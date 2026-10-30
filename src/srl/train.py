"""Train the SRL pipeline on DAP, following `SRL_dynamic` of the SRL codebase.

    uv run python -m src.srl.train --config configs/srl_dap.yaml --out runs/srl/seed_14

Before every `validation.every`-th training episode (every one by default) the
policy is validated on each validation split, and the one with the best mean so
far is saved to <out>/best.pt. The final
policy is saved to <out>/final.pt and one row per episode to <out>/history.csv.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.gym_envs.dap_env import DynamicAssortmentEnvironment
from src.srl.agent import SrlAgent
from src.srl.checkpoint import build_critics, read_checkpoint, save_checkpoint
from src.srl.config import SrlConfig, load_srl_config
from src.srl.evaluate import make_env, mean_revenue
from src.srl.networks import Actor


def initial_actor(
    config: SrlConfig, env: DynamicAssortmentEnvironment, generator: torch.Generator
) -> tuple[Actor, int, float]:
    """First actor drawn from the generator whose mean revenue on the first
    `init.episodes` training seeds lies in `init.reward_range`, with its draw
    index and revenue."""
    init = config.init
    seeds = range(1, init.episodes + 1)
    for draw in range(init.max_draws):
        actor = Actor(generator)
        agent = SrlAgent(config, actor, *build_critics(config, generator))
        reward = mean_revenue(env, agent.best_assortment, seeds, config.env.max_steps)
        if init.reward_range is None:
            return actor, draw, reward
        low, high = init.reward_range
        if low <= reward <= high:
            return actor, draw, reward
    raise RuntimeError(
        f"No initial actor in {init.max_draws} draws earned a mean revenue in "
        f"{init.reward_range}."
    )


def validate(
    agent: SrlAgent, env: DynamicAssortmentEnvironment, config: SrlConfig
) -> dict[str, float]:
    episodes = config.validation.episodes
    return {
        split: mean_revenue(
            env,
            agent.best_assortment,
            range(first_seed, first_seed + episodes),
            config.env.max_steps,
        )
        for split, first_seed in config.validation.splits.items()
    }


def train(config: SrlConfig, out_dir: Path) -> Path:
    """Train with `config`, writing to out_dir. Returns the best checkpoint."""
    torch.set_num_threads(config.torch_threads)
    start = time.perf_counter()
    env = make_env(config.env)
    validation_env = make_env(config.env)

    generator = torch.Generator().manual_seed(config.seed)
    actor, init_draw, init_reward = initial_actor(config, validation_env, generator)
    agent = SrlAgent(config, actor, *build_critics(config, generator))
    tqdm.write(f"Initial actor: draw {init_draw}, mean revenue {init_reward:.1f}")
    init_info = {"init_draw": init_draw, "init_reward": init_reward}

    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    best_score = None
    best_episode = None
    history_path = out_dir / "history.csv"
    with history_path.open("w", newline="") as history_file:
        writer = None
        for episode in tqdm(range(1, config.episodes + 1), desc="Episodes"):
            # Episodes without validation leave its columns empty.
            validated = (episode - 1) % config.validation.every == 0
            if validated:
                rewards = validate(agent, validation_env, config)
                score = sum(rewards.values()) / len(rewards)
            else:
                rewards = dict.fromkeys(config.validation.splits)
                score = None
            # Ties keep the later model, as `reward_comparison` does.
            is_best = validated and (best_score is None or score >= best_score)
            if is_best:
                best_score, best_episode = score, episode
                save_checkpoint(
                    best_path, agent, episode=episode, score=score, **init_info
                )

            episode_seed = episode % config.env.n_train_seeds + 1
            row = {
                "episode": episode,
                **{f"{split}_reward": reward for split, reward in rewards.items()},
                "score": score,
                "best": is_best,
                "episode_seed": episode_seed,
                "sigma_f": agent.sigma_f,
                "sigma_b": agent.sigma_b,
                "temperature": agent.temperature,
                "lr": agent.lr,
            }
            transitions = agent.collect_episode(env, episode_seed)
            row["episode_revenue"] = sum(t.revenue for t in transitions)
            row |= agent.update(transitions)
            agent.end_episode()
            row["elapsed_s"] = time.perf_counter() - start

            if writer is None:
                writer = csv.DictWriter(history_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            history_file.flush()
            if validated:
                tqdm.write(
                    f"episode {episode}: "
                    + ", ".join(f"{s} {r:.1f}" for s, r in rewards.items())
                    + (" (best)" if is_best else "")
                )

    train_time = time.perf_counter() - start
    save_checkpoint(
        out_dir / "final.pt",
        agent,
        episode=config.episodes + 1,
        train_time_s=train_time,
        **init_info,
    )
    # The best checkpoint also records the time of the whole training run.
    best = read_checkpoint(best_path)
    best["train_time_s"] = train_time
    torch.save(best, best_path)
    tqdm.write(
        f"Best validation score {best_score:.1f} before episode {best_episode}; "
        f"training took {train_time / 60:.1f} min."
    )
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train SRL on DAP.")
    parser.add_argument("--config", type=Path, default=Path("configs/srl_dap.yaml"))
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--seed", type=int, help="Overrides the config's seed.")
    args = parser.parse_args()

    config = load_srl_config(args.config)
    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )
    train(config, args.out)


if __name__ == "__main__":
    main()
