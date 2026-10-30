# Handover: SRL port and same-machine comparison (updated 2026-10-30)

Plan: [srl_port_plan.md](srl_port_plan.md). The SRL algorithm of Hoppe et al.
(NeurIPS 2025) is ported for DAP to the standalone package `src/srl/`, which
uses our DAP environment. The port is verified (phases 1-3). The comparison
tooling for phase 4 is ready. **Waiting on:** the MILP-PPO (CORL) runs on the
server, see [Server runs](#server-runs).

## State of the branch

- Branch `srl-port`, created from `main` at `1b98576`. No pull request yet.
- New files: `src/srl/`, `tests/`, `configs/srl_dap.yaml`,
  `configs/srl_dap_val10.yaml`, `scripts/run_srl_port.sh`,
  `docs/srl_port_plan.md`, this file.
- Changed files:
  - `compare_srl_paper.py`: `--srl`, a `train min` column and `--paired`.
  - `scripts/run_srl_comparison.sh`: writes `train_time_s` per seed.
  - `pyproject.toml` and `uv.lock`: pytest as a dev dependency, and the
    pytest config.
- `train.py`, the `Model`/`Solver` interfaces and `src/config/` are unchanged.
  `src/srl` imports only `src/gym_envs/dap_env.py` and `run_episode` from
  `src/utils/evaluation.py`.
- The results are in `runs/`, which is gitignored, so they exist only on the
  local machine where they ran, in the session worktree
  `.claude/worktrees/srl-porting-feasibility-c63c3a/runs/`:
  - `runs/srl_port_10seeds/`: the 10 paper seeds with validation every
    episode. One directory per seed with `best.pt`, `final.pt`,
    `history.csv` and `train.log`, plus `comparison.txt` and
    `comparison.csv`.
  - `runs/srl_single/seed_14/`: the first single-seed run, identical in
    setup to seed 14 above.
  - `runs/srl_checks/sil_seed14.log`: the SIL check.

## What is there

| File | Contents |
|---|---|
| `src/srl/networks.py` | Actor (linear 10→5→1 per item), reward critic, future critic. Glorot-uniform weights and zero biases, as Flux. |
| `src/srl/co_layer.py` | Top-K, in torch and numpy |
| `src/srl/fenchel_young.py` | Perturbed Fenchel-Young loss (ε = 1, 20 samples) |
| `src/srl/target.py` | Softmax target action over the perturbed candidates |
| `src/srl/buffer.py` | Transitions, returns, replay buffer |
| `src/srl/config.py` | `SrlConfig` (pydantic). Defaults are 03_SRL.jl and Table 7 of the paper. `validation.every` (default 1) sets how often to validate. |
| `src/srl/agent.py` | `SrlAgent`: exploration episode, update step, schedules |
| `src/srl/train.py` | Training entry point: validation, `best.pt`, `final.pt`, `history.csv` |
| `src/srl/evaluate.py` | `evaluate_srl` (rows like `evaluate_params`), `make_env`, and a CLI |
| `src/srl/checkpoint.py` | `torch.save` of the networks, config and metadata |
| `src/srl/checks.py` | `sil` and `critic` checks |
| `tests/test_srl_components.py`, `tests/test_srl_training.py` | 22 tests |

Commands:

```bash
uv run pytest
uv run python -m src.srl.train --config configs/srl_dap.yaml --out runs/srl/seed_14 --seed 14
bash scripts/run_srl_port.sh                                       # 10 paper seeds in parallel + comparison
CONFIG=configs/srl_dap_val10.yaml bash scripts/run_srl_port.sh     # same, validating every 10th episode
uv run python compare_srl_paper.py --srl runs/srl_port_10seeds/seed_*/best.pt --average
uv run python -m src.srl.checks sil
uv run python -m src.srl.checks critic
```

`compare_srl_paper.py` options:
- `params ...`: MILP params files.
- `--srl ...`: SRL checkpoints.
- `--average`: per-episode mean over the seeds, separately for MILP and SRL.
- `--paired A B`: the mean per-episode revenue difference A − B with a 95 %
  bootstrap CI over episodes and model seeds. With `--average`, the MILP and
  SRL averages are paired by default.
- `train min` column: from `train_time_s` next to a params file, or inside
  an SRL checkpoint.

Run time on the local machine (16 cores):
- **SRL:** 8.4 min for one run alone, 13 min each with all 10 in parallel. Each
  run uses one thread. The paper reports 31 min on a MacBook M3 in Julia.
- **CORL** with `configs/dap_srl.yaml` (`scip_brute`):
  - one training iteration (an episode plus the PPO update) takes 30-70 s,
  - validation takes 5-8 s per episode, 80 SCIP solves each,
  - so about 5 h per seed with `validation.every: 10`, and 22-30 h with
    `every: 1`.

## Choices in the port that are easy to get wrong

Each of these follows the Julia source (`scripts/DAP/03_SRL.jl` and
`scripts/DAP/utils/utils.jl`):

- **σ_f and σ_b are variances.** The SRL code samples from `MvNormal(θ, σ * I)`,
  whose covariance is σI, so the noise standard deviation is `sqrt(σ)`. The
  paper calls σ a standard deviation.
- **The 40 target perturbations are shared across a batch.** They come from
  `MersenneTwister(e * i)` in Julia, and from `default_rng(e * i)` here.
- **Exploration noise** is seeded with episode seed × step.
- **The returns keep the upstream quirk:** the last step's return is 0, and the
  second-to-last step's return is its own revenue only.
- **Optimizers.** Gradients are clipped elementwise to ±1e-3 before Adam. The
  learning rate falls linearly from 1e-3 to 5e-4, and both critics follow the
  actor's rate.
- **The future critic** gets 50 shuffled passes over each new episode, with a
  fixed shuffle seed every episode (as Julia's `DataLoader(rng=MersenneTwister(0))`)
  and incomplete batches dropped.
- **Training seeds.** Episode `e` uses seed `e % 100 + 1`. The best model is
  the one with the best mean over train seeds 1-30 and validation seeds
  1001-1030, and ties keep the later model.
- **Initial actor.** It is redrawn until its mean revenue on seeds 1-10 is in
  100-200, the criterion in `00_setup.jl`. The draw used is in each
  `train.log` (between 0 and 10 for the paper seeds).
- **Numbers differ from Julia's,** so results do not match it number for number:
  - the actor is evaluated as one fused linear map in numpy, which is the same
    function in float64,
  - the random number generators differ,
  - the noise of the Fenchel-Young loss comes from its own generator, seeded
    from the model seed.

## Verification results

| Check | Result |
|---|---|
| Gate 1: unit tests | 22 pass. Shapes, networks checked against hand-computed outputs, top-K. The training loss: its gradient against the formula and finite differences, its lower bound, about 0 at a dominant target. The target action, the returns quirk, the buffer wrap-around, collected episodes replaying in the environment, and `validation.every`. |
| Gate 2: determinism | Two runs give identical `history.csv`, and a reloaded `best.pt` reproduces its validation score (in the tests). |
| Critic check (seed 14) | Held-out Huber loss 13.4 → 1.1. Spearman correlation with the true expected revenue 0.16 → 0.27 pooled, 0.15 → 0.24 within a state. Positive but weak, as expected from a linear critic trained on single noisy purchases. |
| SIL check (seed 14) | +23.2 % / +13.2 % over greedy, 74 % / 79 % of the gap from greedy to expert (train / test). The paper's SIL reaches +11.7 % / +7.3 %. |
| One seed (14) | Validation goes 220 → about 450 within 10 episodes, then about 470-500. Best at episode 159. +19.3 % / +11.2 % over greedy. |
| Ten seeds | See below |

Ten paper seeds (`runs/srl_port_10seeds/comparison.txt`), mean over the seeds
of the per-episode revenue:

| | train rev | train gain over greedy | train share of expert gap | test rev | test gain over greedy | test share of expert gap |
|---|---|---|---|---|---|---|
| paper SRL | 529.8 | +20.5 % | 69.2 % | 555.3 | +14.7 % | 71.8 % |
| our port | 514.5 | +21.1 % | 67.6 % | 545.0 | +11.2 % | 67.2 % |
| our greedy / expert | 424.9 / 557.5 | | | 490.2 / 571.8 | | |

Paired against greedy, the SRL average gains +89.6 (train) and +54.8 (test)
revenue per episode, with 95 % CIs of +64 to +117 and +31 to +80.

Gate 3 criteria:

- **Mean share of the expert gap within ±10 points of the paper: met**
  (67.6 vs 69.2, 67.2 vs 71.8).
- **At least 9 of 10 seeds beat greedy on test: met.** All 10 do; the weakest
  is seed 41 at +6.6 %.
- **Spread across seeds of the same order as the paper's Table 1 (1.8 / 1.9):
  accepted with a caveat.**
  - The released SRL scripts compute no standard deviation (`04_plots.jl`
    only plots), so the paper's measure cannot be reproduced exactly.
  - The DAP plots scale rewards by 0.1 ("val. rew. (10^1)"). If Table 1 uses
    the same scale, our values bracket the paper's:
    - the standard deviation of the 10 runs' mean revenue is 0.83 / 0.89,
    - the mean per-episode standard deviation across runs is 3.6 / 2.7.
  - Unscaled, the first measure is 8.3 / 8.9, about 4-5× the paper's value.
  - The seeds do differ for real: seed 16 beats seed 41 by +30.1 revenue per
    test episode (95 % CI +18.9 to +42.2). The port is accepted on the mean
    results, with this difference noted.

Other observations:
- The mean best-so-far validation score is 354, 416, 469, 487, 494 and 498 at
  episodes 10, 25, 50, 100, 150 and 200, so learning mostly finishes by about
  episode 100, as in Figure 5 of the paper.
- 5 of the 10 seeds found their best model before episode 60.

## Server runs

The comparison needs MILP-PPO (CORL) runs, which are too slow for the local machine.
SRL is rerun on the same server so that the training times are comparable.

1. **Choose the validation protocol.** It only decides which checkpoint is
   kept as best, and must be the same for both algorithms.
   - **Every 10th episode (recommended):** about 5 h per CORL seed. Use
     `configs/dap_srl.yaml` as it is (`validation.every: 10`) and
     `configs/srl_dap_val10.yaml` for SRL.
   - **Every episode,** the paper's protocol: about 22-30 h per CORL seed.
     Set `validation.every: 1` in `configs/dap_srl.yaml`, and use
     `configs/srl_dap.yaml` for SRL.
2. **On the server**, with `srl-port` pulled, run the CORL seeds (10 in
   parallel needs at least 10 cores; set `JOBS` otherwise):

   ```bash
   WANDB_MODE=offline RUN_DIR=runs/corl_dap_val10 nohup bash scripts/run_srl_comparison.sh > corl.log 2>&1 &
   ```

3. **When that has finished,** run SRL. Not at the same time, so the two do
   not compete for the CPU:

   ```bash
   CONFIG=configs/srl_dap_val10.yaml RUN_DIR=runs/srl_port_val10 nohup bash scripts/run_srl_port.sh > srl.log 2>&1 &
   ```

4. **Copy back** `runs/corl_dap_val10/` and `runs/srl_port_val10/`. Each CORL
   seed directory needs its `*_best.yaml` and `train_time_s`, and each SRL
   seed directory its `best.pt`.

## Next steps

1. **The final comparison**, when the server runs are back:

   ```bash
   uv run python compare_srl_paper.py runs/corl_dap_val10/seed_*/*_best.yaml \
       --srl runs/srl_port_val10/seed_*/best.pt --average \
       --csv runs/final_comparison.csv
   ```

   This prints both algorithms next to the paper, with training time and the
   paired difference between the CORL and SRL averages. Write up the results
   here.
2. **Optional:** SRL's training on our `DapModel` restricted to the
   immediate-score term (`w = 0`, no value pieces), which is the same policy
   class as SRL's actor. Also the plan's ablations (`dynamic: false`, no
   clipping), if a result needs explaining.
