# Handover: SRL port and same-machine comparison (2026-10-29)

Plan: [srl_port_plan.md](srl_port_plan.md). The SRL algorithm of Hoppe et al.
(NeurIPS 2025) is ported for DAP to the standalone package `src/srl/`, which
uses our DAP environment. Phases 1-3 of the plan are done: the port passes
every check except one open question about the spread across seeds. Phase 4
(the comparison with our MILP-PPO) has not started.

## State of the branch

- Branch `srl-port`, created from `main` at `1b98576`, committed and pushed
  to `origin/srl-port`. No pull request yet.
- New files: `src/srl/`, `tests/`, `configs/srl_dap.yaml`,
  `scripts/run_srl_port.sh`, `docs/srl_port_plan.md`, this file.
- Changed files:
  - `compare_srl_paper.py`: new `--srl` option.
  - `pyproject.toml` and `uv.lock`: pytest as a dev dependency, and the
    pytest config.
- `train.py`, the `Model`/`Solver` interfaces and `src/config/` are unchanged.
  `src/srl` imports only `src/gym_envs/dap_env.py` and `run_episode` from
  `src/utils/evaluation.py`.
- The results are in `runs/`, which is gitignored, so they exist only on this
  machine:
  - `runs/srl_port_10seeds/`: the 10 paper seeds. One directory per seed
    with `best.pt`, `final.pt`, `history.csv` and `train.log`, plus
    `comparison.txt` and `comparison.csv`.
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
| `src/srl/config.py` | `SrlConfig` (pydantic). Defaults are 03_SRL.jl and Table 7 of the paper. |
| `src/srl/agent.py` | `SrlAgent`: exploration episode, update step, schedules |
| `src/srl/train.py` | Training entry point: validation before every episode, `best.pt`, `final.pt`, `history.csv` |
| `src/srl/evaluate.py` | `evaluate_srl` (rows like `evaluate_params`), `make_env`, and a CLI |
| `src/srl/checkpoint.py` | `torch.save` of the networks, config and metadata |
| `src/srl/checks.py` | `sil` and `critic` checks |
| `tests/test_srl_components.py`, `tests/test_srl_training.py` | 21 tests |

Commands:

```bash
uv run pytest
uv run python -m src.srl.train --config configs/srl_dap.yaml --out runs/srl/seed_14 --seed 14
bash scripts/run_srl_port.sh            # 10 paper seeds in parallel + comparison
uv run python compare_srl_paper.py --srl runs/srl_port_10seeds/seed_*/best.pt --average
uv run python -m src.srl.checks sil
uv run python -m src.srl.checks critic
```

Run time: 8.4 min for one run alone, and 13 min each with all 10 in parallel.
Each run uses one thread. The paper reports 31 min on a MacBook M3 in Julia.

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
| Gate 1: unit tests | 21 pass. Shapes, networks checked against hand-computed outputs, top-K. The training loss: its gradient against the formula and finite differences, its lower bound, about 0 at a dominant target. The target action, the returns quirk, the buffer wrap-around, and collected episodes replaying in the environment. |
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

Gate 3 criteria:

- **Mean share of the expert gap within ±10 points of the paper: met**
  (67.6 vs 69.2, 67.2 vs 71.8).
- **At least 9 of 10 seeds beat greedy on test: met.** All 10 do; the weakest
  is seed 41 at +6.6 %.
- **Spread across seeds of the same order as the paper's Table 1 (1.8 / 1.9):
  open.**
  - Our standard deviation of the 10 runs' mean revenue is 8.3 on train
    and 8.9 on test.
  - The mean per-episode standard deviation across runs is 36 / 27.
  - The paper does not define its measure precisely. "Mean of the standard
    deviations across the validation and final test rewards" matches none of
    these.
  - Either the paper's measure is a different quantity, or our runs vary about
    4-5× more between seeds. Seed 41 on test accounts for part of it.

Other observations:
- The mean best-so-far validation score is 354, 416, 469, 487, 494 and 498 at
  episodes 10, 25, 50, 100, 150 and 200, so learning mostly finishes by about
  episode 100, as in Figure 5 of the paper.
- 5 of the 10 seeds found their best model before episode 60.

## Next steps

1. **Settle the open spread criterion.** Options:
   - work out the paper's measure (Table 1 caption, Appendix C.1.4),
   - accept the port as validated on the mean results and note the
     difference,
   - run the Julia code for a few seeds to compare spreads (Julia 1.11.5 is
     installed through juliaup; the plan's fallback section has the risks).

   Cheap diagnostic: compare the `history.csv` curves of seed 41 with the
   others.
2. **Phase 4, the MILP-PPO side.** Run `scripts/run_srl_comparison.sh` (it
   trains our MILP model with PPO). Set `validation.every: 1` in
   `configs/dap_srl.yaml` first, so both algorithms use the same protocol
   (it is 10 now). It uses `scip_brute` and is much slower than SRL; check
   the time per run before starting all 10 seeds.
3. **Compare both in one table:**

   ```bash
   uv run python compare_srl_paper.py runs/srl_dap_<timestamp>/seed_*/*_best.yaml \
       --srl runs/srl_port_10seeds/seed_*/best.pt --average
   ```

4. **Extend `compare_srl_paper.py`** with a training-time column
   (`train_time_s` is in every SRL `best.pt`; MILP-PPO needs its own
   timing), and a paired comparison (per-episode revenue difference with a
   bootstrap confidence interval).
5. **Optional:** SRL's training on our `DapModel` restricted to the
   immediate-score term (`w = 0`, no value pieces), which is the same policy
   class as SRL's actor. Also the plan's ablations (`dynamic: false`, no
   clipping) if a gate needs debugging.

Small cleanups: `src/srl/checks.py` calls the agent's private `_step` and
resets `_linear_actor`. Make these public on `SrlAgent` (e.g.
`optimizer_step` and `actor_updated`).
