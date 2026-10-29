# Plan: port SRL to this repo and compare on the same machine

Status and results so far: [srl_port_handover.md](srl_port_handover.md).

Goal: run the comparison of `compare_srl_paper.py` with SRL results we produce
ourselves, on our DAP environment and on this machine, instead of only quoting
Table 11 of Hoppe et al. (NeurIPS 2025). SRL must be shown to work as in the
paper before any comparison run counts.

Source: <https://github.com/tumBAIS/Structured-RL>, Julia 1.11.5. For DAP the
algorithm is `scripts/DAP/03_SRL.jl` (training loop) and
`scripts/DAP/utils/utils.jl` (environment, critics, episode generation, replay
buffer), about 200 lines in total. We port it from the source; the Julia code
is not run.

## Starting point

- `src/gym_envs/dap_env.py` already mirrors the SRL DAP environment, apart from the
  random number generator (numpy instead of `MersenneTwister`).
- Baselines on the paper's seeds, from this repo (`compare_srl_paper.py`, 27 s):

  | split | greedy (ours / paper) | expert (ours / paper) | expert over greedy (ours / paper) |
  |---|---|---|---|
  | train 1-100 | 424.9 / 439.6 | 557.5 / 569.9 | +31.2 % / +29.6 % |
  | test 2001-2100 | 490.2 / 484.1 | 571.8 / 583.2 | +16.7 % / +20.5 % |

  The environment is on the same scale as the paper's. Only its instances differ.
- The paper's SRL on DAP: train 529.8, test 555.3, which is +20.5 % / +14.7 %
  over greedy and 69 % / 72 % of the gap from greedy to expert. The spread
  across 10 seeds is 1.8 / 1.9, and training takes 31 min on a MacBook M3. It
  levels off after about 100-150 episodes (Figure 5).

## Module layout

SRL is a standalone package. It does not touch `train.py`, the `Model`/`Solver`
interfaces or `src/config/config_models.py`. Its only dependencies on the rest of
the repo are the environment and the generic episode runner:

- `src/gym_envs/dap_env.py`: `DynamicAssortmentEnvironment` and
  `DynamicAssortmentSimulator` for training and evaluation, and the
  greedy and expert policies for the sanity checks.
- `src/utils/evaluation.py`: `run_episode(env, act, seed, max_steps)`, so SRL
  episodes are seeded and summed exactly like the other policies.

```
src/srl/
  networks.py       actor, reward critic, future critic
  co_layer.py       top-K
  fenchel_young.py  perturbed Fenchel-Young loss
  target.py         softmax target action from perturbed candidates
  buffer.py         replay buffer, episode returns
  config.py         SrlConfig (pydantic), loaded from YAML
  agent.py          SrlAgent: act, collect_episode, update, schedules
  checkpoint.py     save/load of actor, critics, config, curve, timing
  train.py          entry point: uv run python -m src.srl.train --config ...
  evaluate.py       evaluate_srl(checkpoint, seeds) -> rows as evaluate_params
configs/srl_dap.yaml
scripts/run_srl_port.sh   trains the 10 seeds in parallel
tests/test_srl_*.py
```

The one change outside the package is in phase 4: `compare_srl_paper.py` gets
an `--srl` option for SRL checkpoints, which calls `src.srl.evaluate`.

## Phase 1: port the SRL components

In torch:

| File | Contents | Julia original |
|---|---|---|
| `networks.py` | Actor `Linear(10,5) -> Linear(5,1)` per item, no activation. Reward critic `Linear(6,3)` per shown item -> flatten(12) -> `Linear(12,1)`, mean. Future critic `Linear(11,5)` per item -> flatten(100) -> `Linear(100,10)` -> `Linear(10,1)`. Glorot-uniform weights and zero biases, as Flux. | `00_setup.jl`, `critic_reward`, `critic_future` |
| `co_layer.py` | Top-K: the K highest scores get 1 | `DAP_optimization` |
| `fenchel_young.py` | Perturbed-additive Fenchel-Young loss: `F(θ) = mean_k max_a (θ+εZ_k)ᵀa`, loss `F(θ) − θᵀâ`, gradient `mean_k f(θ+εZ_k) − â`, with ε = 1 and 20 samples | InferOpt `PerturbedAdditive` + `FenchelYoungLoss` |
| `target.py` | 40 perturbed scores with σ_b, top-K on each, critic values, softmax with temperature τ, weighted mean of the actions. Fall back to the argmax if the result is NaN. | inner loop of `SRL_dynamic` |
| `buffer.py` | Ring replay buffer of 8000, uniform sampling. Episode returns with γ = 0.99, including upstream's quirk (the last transition keeps return 0 and the one before it gets only its own revenue). | `rb_add`, `rb_sample`, `generate_episode` |
| `agent.py` | Collection, update step and schedules (below) | `SRL_dynamic` |

Details to copy exactly from the Julia source:

- **Optimizers.** Every optimizer step clips each gradient element to ±1e-3
  (`ClipValue`) and then applies Adam (β = 0.9 / 0.999, ε = 1e-8). The learning
  rate decreases linearly from 1e-3 to 5e-4 over 200 episodes, and both
  critics follow the actor's rate.
- **Exploration noise.** σ_f and σ_b both go linearly from 2 to 1, and τ = 1.
- **Updates per episode.** 100 iterations with batch size 4. Each iteration
  does one actor update (the sum of 4 Fenchel-Young losses) and one
  reward-critic update (Huber with δ = 1 against `rev_t`). After that, the
  future critic gets 50 shuffled passes over the episode just collected, with
  target `ret_t − rev_t`.
- **Critic inputs.** The reward critic sees the first 6 feature rows of the
  shown items. The future critic sees all 10 rows plus the assortment
  indicator. Our observation (`item_features`) has the same 10 rows in the same
  order as Julia's `feature_vector`.
- **Training seeds.** Episode `e` uses seed `e % 100 + 1`.

Unit tests (`tests/test_srl_*.py`, add pytest as a dev dependency):

- **Shapes and layer arithmetic.** Actor and critic outputs have the right
  shapes. Set a few weights by hand and check the outputs against values
  computed by hand.
- **Top-K.** Returns exactly K ones at the K largest scores.
- **Fenchel-Young loss.**
  - The loss is ≥ 0 for any target in the convex hull of the actions, and
    close to 0 when θ strongly favours the target.
  - The autograd gradient equals `mean_k f(θ+εZ_k) − â`.
  - With fixed noise samples, it matches finite differences of the
    Monte Carlo loss.
- **Target action.** A convex combination of vertices that sums to K, and
  moves to the argmax as τ → 0.
- **Returns and replay buffer.** The returns quirk and the buffer wrap-around.
- **Environment use.** An episode collected by the agent replays exactly
  through `DynamicAssortmentEnvironment` with the same seed.

**Gate 1:** all unit tests pass, and the code has been reviewed line by line
against `03_SRL.jl` and `utils.jl`.

## Phase 2: training entry point and checkpoints

- `src/srl/train.py` follows `SRL_dynamic`. Before each of the 200 episodes, it
  validates on train seeds 1-30 and validation seeds 1001-1030 and keeps the
  model with the best mean of the two (ties go to the later model). Then it
  collects one episode with σ_f exploration, runs the 100 update iterations
  and the future-critic passes, and steps the schedules.
- Each episode writes a row to a CSV in the run directory: train and
  validation rewards, losses, σ, learning rate and time. wandb logging is
  optional and off by default.
- The checkpoint (`checkpoint.py`, `torch.save`) holds the actor and both
  critics' weights, the config, the best episode, the curve and the training
  time. The run writes `best.pt` and `final.pt`.
- Initial models: for each of the paper's model seeds (14 16 18 19 28 34 41
  57 72 95), seed torch and build the networks. `00_setup.jl` says the initial
  actor's reward should be between 100 and 200 (10 training episodes) to
  avoid chance convergence. If an initialization falls outside that range,
  draw again with a derived seed and record which draw was used.

**Gate 2:** for one seed, two runs give identical CSVs. Loading `best.pt`
through `src.srl.evaluate` reproduces its validation reward.

## Phase 3: show SRL works

Run the checks in order and stop at the first failure.

1. **Actor and loss without a critic (SIL check).** Train the actor with the
   Fenchel-Young loss on expert assortments (`expert_assortment`) from
   training seeds 1-100. It should clearly beat greedy, around the paper's SIL
   result (+11.7 % / +7.3 %). This checks the actor, the CO-layer and the loss
   independently of the critics.
2. **Critic check.** With the policy fixed, the reward critic's loss goes
   down, and its value ranks assortments in the same order as the
   environment's true `expected_revenue` (clearly positive Spearman
   correlation).
3. **One seed.** The validation curve starts at the initial level (100-200),
   rises and levels off within roughly 100-150 episodes, as in Figure 5 of the
   paper. The final gain over greedy should be close to the paper's.
4. **Ten seeds (`scripts/run_srl_port.sh`).** Gate 3 criteria:
   - mean `expert_gap` on train and test within ±10 points of the paper
     (69 % / 72 %),
   - at least 9 of 10 seeds beat greedy on test,
   - spread across seeds of the same order as the paper's Table 1 (1.8 / 1.9).

If a check fails:
- Isolate the cause with ablations: future critic off (the paper's
  `dynamic = false`), no gradient clipping, a fixed τ and fixed σ.
- Compare each component again against the Julia source.
- If that doesn't explain it, running the Julia code as a reference is still
  possible, since Julia 1.11.5 is installed here.

## Phase 4: the comparison

1. `compare_srl_paper.py` takes SRL checkpoints with `--srl best.pt ...`.
   They are evaluated through `src.srl.evaluate` on the same episodes as
   greedy, expert and the MILP params files, and shown as `ours SRL (port)`
   next to `ours` MILP-PPO and the paper's rows. `--average` also averages the
   SRL runs per episode. Add a training-time column, as in the paper's Table 1.
2. Make the comparison fair: both algorithms use the same 10 seeds, 200
   training episodes and the same validation protocol (every episode, so
   `validation.every: 1` for MILP-PPO), with the same thread limits
   (`OMP_NUM_THREADS=1`) on the same machine.
3. Ported SRL and MILP-PPO run on the same instances, so add a paired
   comparison: the per-episode revenue difference with a bootstrap confidence
   interval, next to the means.
4. Optional: SRL's training applied to our `DapModel` restricted to its
   immediate-score term (`w = 0`, no value pieces). This is the same policy
   class as SRL's linear actor, so it separates the effect of the training
   algorithm from the effect of the richer MILP.

## Time

| Phase | Implementation | Runs (wall-clock) |
|---|---|---|
| 1 Components and tests | 1-2 h | minutes |
| 2 Entry point and checkpoints | about 1 h | minutes |
| 3 Verification | 30 min, plus debugging if a gate fails | about 3 h (checks, 1 seed, 10 seeds in parallel) |
| 4 Comparison | about 1 h | 1-2 h (final SRL and MILP-PPO runs) |

The implementation times are for Claude Code doing the work. The main
uncertainty is debugging if a verification gate fails.
