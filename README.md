# OptimizedRL

Combinatorial optimization combined with RL.

This repo contains the source code for my master thesis, completed June 2025.
It is a framework for applying RL over MILP optimization models for operations research.

## Getting Started

See train.py for an example.

## Additional Code

The original thesis has had additional work done. The addition is the KNN
algorithms, example 2 and PPO algorithm.

## Dynamic Assortment Problem (DAP)

A port of the DAP from "Structured Reinforcement Learning for Combinatorial
Decision-Making" (Hoppe et al., NeurIPS 2025), so CORL can be compared with
its SIL/PPO/SRL results. Each of 80 steps shows 4 of 20 items to a customer
who buys at most one (multinomial logit on a hidden linear utility);
purchases change hype and satisfaction, so decisions affect future demand.

- `src/dap/`: environment (port of the paper's `scripts/DAP/utils/utils.jl`),
  greedy and myopic-expert baselines, the MILP and evaluation.
- `src/models/dap_model.py`: the CORL MILP. Learned linear item revenue,
  optional pairwise interaction costs between shown items (McCormick-
  linearised) and J piecewise-linear value pieces; gradients from the LP
  duals (CORL eq. 18); NNS-k completion of fractional B&B nodes.
- The pairwise terms make the LP relaxation fractional in ~95% of states, so
  SCIP (`scip_brute` with presolve, cuts, heuristics, propagation and conflict
  analysis off) builds a real branch-and-bound tree; its leaves are CORL's
  candidates. Without them the LP is almost always integral and B&B stops at
  the root.
- `src/solvers/exact_assortment.py`: all 4845 assortments, i.e. the exact
  CORL softmax (eq. 10), to measure the B&B approximation.

Configs: `config_dap.yaml` (main: J = 2 + pairwise, SCIP), `config_dap_exact.yaml`
(same MILP, exact softmax) and `config_dap_linear.yaml` (J = 0, no pairwise:
paper 02's linear top-k actor).

```bash
uv run python scripts/dap_smoke_test.py                # correctness checks
uv run python scripts/dap_baselines.py                 # greedy and expert
scripts/train_dap_seeds.sh -c config_dap.yaml          # 10 seeds in parallel
uv run python scripts/dap_baselines.py --params params/dap_dap_corl_scip_seed_*_best.yaml
```

Use `tmux` (or `nohup`) on a server so the runs survive a disconnect.
