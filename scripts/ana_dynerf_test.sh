set -eo pipefail

NSYS_PATH=/opt/nvidia/nsight-systems/2024.5.4
NCU_PATH=/opt/nvidia/nsight-compute/2024.3.1
REPOSITORY_PATH=$(git rev-parse --show-toplevel)
OUTPUT_PATH=$REPOSITORY_PATH/output/dynerf
NCU_BIN=$NCU_PATH/ncu

# Prefer using the currently activated conda env's python even when running under sudo/root.
# `sudo` may reset PATH (secure_path), which would otherwise pick system python.
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
else
    PYTHON_BIN="$(command -v python || true)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "ERROR: Cannot find python interpreter. Activate your conda env first." >&2
    exit 1
fi

# Nsight Compute needs elevated privileges on many Linux setups.
# Re-exec as root while preserving the current environment (conda/python paths).
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "==WARNING== Insufficient privileges for Nsight Compute. Re-run with root:"
    echo "sudo -E bash $0 $*"
    exec sudo -E bash "$0" "$@"
fi

generate_target() {
    target="$PYTHON_BIN $REPOSITORY_PATH/render.py \
        --model_path $OUTPUT_PATH/$1 \
        --configs $REPOSITORY_PATH/arguments/dynerf/$1.py \
        --skip_train \
        --skip_test \
        --quiet \
        --render_num 4"
    echo $target
}

kernel_name='regex:"preprocessCUDA'
kernel_name+="|DeviceScanInitKernel"
kernel_name+="|DeviceScanKernel"
kernel_name+="|duplicateWithKeys"
kernel_name+="|DeviceRadixSortHistogramKernel"
kernel_name+="|DeviceRadixSortExclusiveSumKernel"
kernel_name+="|DeviceRadixSortOnesweepKernel"
kernel_name+="|identifyTileRanges"
kernel_name+='|renderCUDA"'

# Per-kernel latency (duration on GPU)
metrics=(
    "gpu__time_duration.sum"
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
)
metrics_csv="$(
    IFS=','
    echo "${metrics[*]}"
)"

SCENE="cut_roasted_beef"
target=$(generate_target $SCENE)

$NCU_BIN \
    --set full \
    --target-processes all \
    -k $kernel_name \
    -f \
    -o $OUTPUT_PATH/$SCENE/analysis \
    $target
$NCU_BIN \
    -i $OUTPUT_PATH/$SCENE/analysis.ncu-rep \
    --page raw \
    --csv \
    --metrics "$metrics_csv" \
    --log-file $OUTPUT_PATH/$SCENE/analysis.csv
# Aggregated view: one row per kernel (invocation count + min/max/avg across launches)
$NCU_BIN \
    -i $OUTPUT_PATH/$SCENE/analysis.ncu-rep \
    --page raw \
    --csv \
    --metrics "$metrics_csv" \
    --print-summary per-kernel \
    --log-file $OUTPUT_PATH/$SCENE/analysis_per_kernel.csv
