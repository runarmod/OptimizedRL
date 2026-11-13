#!/usr/bin/env bash
# Example problem: train CORL with and without the value term v, then test.
#
#   nohup bash scripts/run_example_value_ablation.sh [SEED ...] > example_value.log 2>&1 &
#
# Every model seed (default 1-10) is its own problem instance, and is trained
# once with v and once without. scripts/example_value_ablation.py then tests
# the untrained and best trained parameters of both variants on the same
# episodes and writes summary.txt.
#
# Environment variables:
#   CONFIG      base config (default: configs/example_value_ablation.yaml)
#   TOTAL_ITERS overrides total_iters
#   RUN_DIR     output directory (default: runs/example_value_<timestamp>)
#   EPISODES    test episodes per run (default: 200)
set -euo pipefail
cd "$(dirname "$0")/.."

seeds=("$@")
if (( ${#seeds[@]} == 0 )); then
    seeds=(1 2 3 4 5 6 7 8 9 10)
fi
config=${CONFIG:-configs/example_value_ablation.yaml}
run_dir=${RUN_DIR:-runs/example_value_$(date +%Y%m%d_%H%M%S)}
n_runs=$(( 2 * ${#seeds[@]} ))
cores=$(nproc)
cores_per_run=$(( cores > n_runs ? cores / n_runs : 1 ))

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TQDM_MININTERVAL=60

mkdir -p "$run_dir"
cp "$config" "$run_dir/base_config.yaml"
git rev-parse HEAD > "$run_dir/commit.txt" 2>/dev/null || true
uv sync

train() {
    local seed=$1 variant=$2 value_term=$3
    local dir="$run_dir/seed_$seed/$variant"
    mkdir -p "$dir"
    sed -e "s/^numpy_seed: .*/numpy_seed: $seed/" \
        -e "s/^name: .*/name: \"example value ablation seed $seed $variant\"/" \
        -e "s/^  value_term: .*/  value_term: $value_term/" \
        "$config" > "$dir/config.yaml"
    if [[ -n ${TOTAL_ITERS:-} ]]; then
        sed -i "s/^total_iters: .*/total_iters: $TOTAL_ITERS/" "$dir/config.yaml"
    fi
    echo "$(date '+%F %T') seed $seed $variant started"
    if PYTHON_CPU_COUNT=$cores_per_run uv run python train.py --config "$dir/config.yaml" \
        --params-dir "$dir" > "$dir/train.log" 2>&1; then
        echo "$(date '+%F %T') seed $seed $variant finished"
    else
        echo "$(date '+%F %T') seed $seed $variant FAILED, see $dir/train.log"
    fi
}

echo "Training seeds ${seeds[*]} with and without v into $run_dir"
for seed in "${seeds[@]}"; do
    train "$seed" v true &
    train "$seed" no_v false &
done
wait

echo "$(date '+%F %T') testing"
uv run python scripts/example_value_ablation.py "$run_dir" --episodes "${EPISODES:-200}" \
    2>&1 | tee "$run_dir/test.txt"
echo "$(date '+%F %T') done, results in $run_dir/summary.txt"
