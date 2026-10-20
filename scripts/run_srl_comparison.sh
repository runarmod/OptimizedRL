#!/usr/bin/env bash
# Train the DAP model with the SRL paper's protocol and compare with the paper.
#
#   bash scripts/run_srl_comparison.sh [SEED ...]
#
# Trains one run per model seed (default: the ten seeds of the paper) in
# parallel, then evaluates the best model of every run on the paper's train
# and test episodes with compare_srl_paper.py. Everything is written to
# runs/srl_dap_<timestamp>/. Start it under nohup or tmux so it survives
# logging out of the server:
#
#   nohup bash scripts/run_srl_comparison.sh > srl_comparison.log 2>&1 &
#
# Environment variables:
#   JOBS        runs trained at the same time (default: one per seed)
#   CONFIG      config every run starts from (default: configs/dap_srl.yaml)
#   WANDB_MODE  overrides wandb_mode of the config, e.g. offline
#   RUN_DIR     output directory (default: runs/srl_dap_<timestamp>)
#   EPISODES    comparison episodes per split (default: 100, as the paper)
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."

seeds=("$@")
if (( ${#seeds[@]} == 0 )); then
    # Model seeds of the SRL paper (scripts/DAP/00_setup.jl of its codebase).
    seeds=(14 16 18 19 28 34 41 57 72 95)
fi
config=${CONFIG:-configs/dap_srl.yaml}
max_jobs=${JOBS:-${#seeds[@]}}
run_dir=${RUN_DIR:-runs/srl_dap_$(date +%Y%m%d_%H%M%S)}

# One core per run: SCIP is single-threaded and the networks are small.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Keep the progress bars in the logs short.
export TQDM_MININTERVAL=60

mkdir -p "$run_dir"
cp "$config" "$run_dir/base_config.yaml"
git rev-parse HEAD > "$run_dir/commit.txt" 2>/dev/null || true
uv sync

train_seed() {
    local seed=$1
    local seed_dir="$run_dir/seed_$seed"
    mkdir -p "$seed_dir"
    sed -e "s/^numpy_seed: .*/numpy_seed: $seed/" \
        -e "s/^name: .*/name: \"dap srl-comparison seed $seed\"/" \
        "$config" > "$seed_dir/config.yaml"
    if [[ -n ${WANDB_MODE:-} ]]; then
        sed -i "s/^wandb_mode: .*/wandb_mode: \"$WANDB_MODE\"/" "$seed_dir/config.yaml"
    fi

    echo "$(date '+%F %T') seed $seed started"
    if uv run python train.py --config "$seed_dir/config.yaml" \
        --params-dir "$seed_dir" > "$seed_dir/train.log" 2>&1; then
        touch "$seed_dir/finished"
        echo "$(date '+%F %T') seed $seed finished"
    else
        echo "$(date '+%F %T') seed $seed FAILED, see $seed_dir/train.log"
    fi
}

echo "Training seeds ${seeds[*]} ($max_jobs at a time) into $run_dir"
for seed in "${seeds[@]}"; do
    while (( $(jobs -rp | wc -l) >= max_jobs )); do
        wait -n
    done
    train_seed "$seed" &
done
wait

# Test the best model of every finished run, as SRL does, or its final model
# if it did not validate during training (validation.every: 0).
trained=()
for seed in "${seeds[@]}"; do
    seed_dir="$run_dir/seed_$seed"
    if [[ ! -e $seed_dir/finished ]]; then
        echo "seed $seed did not finish, leaving it out of the comparison"
        continue
    fi
    best=("$seed_dir"/*_best.yaml)
    if (( ${#best[@]} > 0 )); then
        trained+=("${best[0]}")
        continue
    fi
    for file in "$seed_dir"/*.yaml; do
        if [[ $(basename "$file") != config.yaml ]]; then
            trained+=("$file")
        fi
    done
done
if (( ${#trained[@]} == 0 )); then
    echo "No run finished, nothing to compare."
    exit 1
fi

compare_args=(--episodes "${EPISODES:-100}" --csv "$run_dir/comparison.csv")
if (( ${#trained[@]} > 1 )); then
    compare_args+=(--average)
fi
echo "$(date '+%F %T') comparing ${#trained[@]} runs with the paper"
uv run python compare_srl_paper.py "${trained[@]}" "${compare_args[@]}" \
    2>&1 | tee "$run_dir/comparison.txt"
echo "$(date '+%F %T') done, results in $run_dir/comparison.txt"
