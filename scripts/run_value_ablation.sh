#!/usr/bin/env bash
# Value-term ablation: train the DAP model with and without the value term v in
# the MILP, 50 training episodes each, on the same model seeds, and compare both
# with the SRL paper's protocol (scripts/run_srl_comparison.sh).
#
#   nohup bash scripts/run_value_ablation.sh > value_ablation.log 2>&1 &
#   nohup bash scripts/run_value_ablation.sh 14 16 18 > value_ablation.log 2>&1 &
#
# Seeds default to the ten seeds of the paper. Results go to
# runs/value_ablation_<timestamp>/{value,no_value}/comparison.txt.
#
# Environment variables (passed on to run_srl_comparison.sh):
#   JOBS        runs trained at the same time per variant (default: one per seed)
#   PARALLEL    1 trains both variants at the same time (default: 0, one after
#               the other; set JOBS so that 2 x JOBS fits the cores)
#   WANDB_MODE  overrides wandb_mode of the configs, e.g. offline
#   EPISODES    comparison episodes per split (default: 100)
set -euo pipefail
cd "$(dirname "$0")/.."

root=runs/value_ablation_$(date +%Y%m%d_%H%M%S)
mkdir -p "$root"

run_variant() {
    local variant=$1
    shift
    CONFIG=configs/dap_ablation_$variant.yaml \
        RUN_DIR=$root/$variant \
        NAME="dap ablation $variant" \
        bash scripts/run_srl_comparison.sh "$@"
}

if [[ ${PARALLEL:-0} == 1 ]]; then
    run_variant value "$@" > "$root/value.log" 2>&1 &
    run_variant no_value "$@" > "$root/no_value.log" 2>&1 &
    wait
else
    run_variant value "$@"
    run_variant no_value "$@"
fi

for variant in value no_value; do
    echo "===== $variant ====="
    cat "$root/$variant/comparison.txt" 2>/dev/null || echo "(no comparison, see logs)"
done
