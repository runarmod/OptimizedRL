# OptimizedRL

Combinatorial optimization combined with RL.

This repo contains the source code for my master thesis, completed June 2025.
It is a framework for applying RL over MILP optimization models for operations research.

## Getting Started

See train.py for an example.

## Additional Code

The original thesis has had additional work done. The addition is the KNN
algorithms, example 2 and PPO algorithm.

## Dynamic Vehicle Scheduling (DVSP)

A port of the DVSP from "Structured Reinforcement Learning for Combinatorial
Decision-Making" (Hoppe et al., NeurIPS 2025), so CORL can be compared with
its SIL/PPO/SRL results. The 30 instances it uses (from the
[EURO-NeurIPS 2022 quickstart](https://github.com/ortec/euro-neurips-vrp-2022-quickstart))
are included in `data/euro_neurips_2022/`, so the repo is self-contained.

- `src/dvsp/`: instance parsing, simulator, features, prize-collecting LP,
  anticipative expert and evaluation, ported from DynamicVehicleRouting.jl
  (commit 0af4301, the version pinned by paper 02).
- `src/models/dvsp_model.py`: the MILP policy. Prizes theta_v = w . phi_v
  (14 features, as paper 02's actor); CORL learns w.
- `src/gym_envs/dvsp_env.py`: training environment; actions are arc vectors.
- `src/solvers/enumeration.py`: candidate pool by best-first branching on the
  dispatch decisions. The DVSP LP relaxation is integral, so SCIP's B&B tree
  has only a root and cannot supply the candidates CORL's policy samples from.

```bash
uv run python scripts/dvsp_smoke_test.py         # correctness checks
uv run python scripts/dvsp_check_integrality.py  # LP integrality / B&B collapse
uv run python scripts/dvsp_baselines.py          # greedy, lazy, expert
uv run python train.py --config config_dvsp.yaml # train CORL-PPO
uv run python scripts/dvsp_baselines.py --params params/dvsp_<name>_best.yaml
```

### Running the 10 seeds on a server

Needs [uv](https://docs.astral.sh/uv/) (it installs Python 3.14 and the
dependencies) and, for the default `wandb_mode: online`, `uv run wandb login`.

```bash
git clone -b dvsp-environment git@github.com:runarmod/OptimizedRL.git
cd OptimizedRL
scripts/train_dvsp_seeds.sh        # seeds 0-9, all in parallel (~20 min)
scripts/train_dvsp_seeds.sh -j 5   # at most 5 at a time (each ~1 core, 0.5 GB)
```

Weights are saved to `params/dvsp_dvsp_corl_ppo_seed_<k>_best.yaml`, logs to
`logs/dvsp_seed<k>.out`.

Paper 02's RL evaluation stops before the final epoch, while its greedy and
expert baselines also dispatch it; `dvsp.serve_final_epoch` selects the
protocol and `dvsp_baselines.py` reports both.
