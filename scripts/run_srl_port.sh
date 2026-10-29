#!/usr/bin/env bash
# Train our SRL port (src/srl) on DAP and compare it with the paper.
#
#   bash scripts/run_srl_port.sh [SEED ...]
#
# Trains one run per model seed (default: the ten seeds of the paper) in
# parallel, then evaluates the best model of every run on the paper's train
# and test episodes with compare_srl_paper.py --srl. Everything is written to
# runs/srl_port_<timestamp>/. Start it under nohup or tmux so it survives
# logging out of the server:
#
#   nohup bash scripts/run_srl_port.sh > srl_port.log 2>&1 &
#
# Environment variables:
#   JOBS      runs trained at the same time (default: one per seed)
#   CONFIG    config every run starts from (default: configs/srl_dap.yaml)
#   RUN_DIR   output directory (default: runs/srl_port_<timestamp>)
#   EPISODES  comparison episodes per split (default: 100, as the paper)
set -euo pipefail
cd "$(dirname "$0")/.."

seeds=("$@")
if (( ${#seeds[@]} == 0 )); then
    # Model seeds of the SRL paper (scripts/DAP/00_setup.jl of its codebase).
    seeds=(14 16 18 19 28 34 41 57 72 95)
fi
config=${CONFIG:-configs/srl_dap.yaml}
max_jobs=${JOBS:-${#seeds[@]}}
run_dir=${RUN_DIR:-runs/srl_port_$(date +%Y%m%d_%H%M%S)}

# One core per run: the networks are small.
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
    echo "$(date '+%F %T') seed $seed started"
    if uv run python -m src.srl.train --config "$config" --out "$seed_dir" \
        --seed "$seed" > "$seed_dir/train.log" 2>&1; then
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

# A run is finished when it wrote final.pt; test its best model, as SRL does.
trained=()
for seed in "${seeds[@]}"; do
    seed_dir="$run_dir/seed_$seed"
    if [[ -e $seed_dir/final.pt ]]; then
        trained+=("$seed_dir/best.pt")
    else
        echo "seed $seed did not finish, leaving it out of the comparison"
    fi
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
uv run python compare_srl_paper.py --srl "${trained[@]}" "${compare_args[@]}" \
    2>&1 | tee "$run_dir/comparison.txt"
echo "$(date '+%F %T') done, results in $run_dir/comparison.txt"
