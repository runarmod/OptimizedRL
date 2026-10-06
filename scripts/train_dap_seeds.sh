#!/usr/bin/env bash
# Train CORL on the DAP with several seeds in parallel.
#
# Each seed runs train.py with the given config, only changing `numpy_seed` and
# appending " seed <k>" to `name`, so parameters are saved per seed (for
# config_dap.yaml: params/dap_dap_corl_scip_seed_<k>_best.yaml). Logs go to
# logs/<config>_seed<k>.out.
#
# Usage: scripts/train_dap_seeds.sh [-c config_dap.yaml] [-s "0 1 2 3 4 5 6 7 8 9"] [-j 10]
#   -c  config (config_dap.yaml, config_dap_exact.yaml or config_dap_linear.yaml)
#   -s  seeds (default 0..9)
#   -j  parallel runs (default: number of seeds); each uses about one core
set -euo pipefail

cd "$(dirname "$0")/.."

config="config_dap.yaml"
seeds="0 1 2 3 4 5 6 7 8 9"
jobs=""
while getopts "c:s:j:" opt; do
    case "$opt" in
        c) config="$OPTARG" ;;
        s) seeds="$OPTARG" ;;
        j) jobs="$OPTARG" ;;
        *) sed -n '2,13p' "$0"; exit 1 ;;
    esac
done
jobs="${jobs:-$(wc -w <<< "$seeds")}"

mkdir -p logs
# One thread per run, so parallel runs do not compete for cores.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

run_seed() {
    local seed="$1" config="$2" cfg start code log
    cfg="$(mktemp --suffix=.yaml)"
    log="logs/$(basename "$config" .yaml)_seed$seed.out"
    sed -e "s/^name: \"\(.*\)\"/name: \"\1 seed $seed\"/" \
        -e "s/^numpy_seed: .*/numpy_seed: $seed/" "$config" > "$cfg"
    start="$(date +%H:%M)"
    set +e
    uv run python train.py --config "$cfg" > "$log" 2>&1
    code=$?
    set -e
    rm -f "$cfg"
    echo "seed $seed finished with exit $code ($start - $(date +%H:%M)), log: $log"
}
export -f run_seed

echo "Training seeds [$seeds] with $jobs parallel run(s), config $config"
uv sync --quiet
tr ' ' '\n' <<< "$seeds" | grep -v '^$' | xargs -P "$jobs" -I{} bash -c 'run_seed "$1" "$2"' _ {} "$config"
echo "All seeds done. Evaluate with: uv run python scripts/dap_baselines.py --params params/dap_*_seed_*_best.yaml"
