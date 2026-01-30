#!/bin/bash
# DyNeRF Sensitivity Analysis - Run analysis for all scenes and aggregate results

SCENES=(
    "aleks-teapot"
    "cut-lemon1"
    "hand1-dense-v2"
    "interp-chicken"
    "slice-banana"
    "torchocolate"
)

DATASET="hypernerf/interp"
MODE="ratio"
RATIO_STEP=5
NUM_LAMBDAS=20
NUM_TIME_SAMPLES=60
METRICS="psnr,ssim,lpips"
LPIPS_NET="vgg"

echo "=========================================="
echo "DyNeRF Sensitivity Analysis - All Scenes"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Mode: $MODE"
echo "Static Ratio Step: $RATIO_STEP%"
echo "Metrics: $METRICS"
echo "Scenes: ${SCENES[@]}"
echo "=========================================="

# Run analysis for each scene
for scene in "${SCENES[@]}"; do
    echo ""
    echo "Processing scene: $scene"
    echo "----------------------------------------"
    bash ./scripts/run_sensitivity_analysis.sh "$scene" "$DATASET" "$MODE" "$RATIO_STEP" "$NUM_LAMBDAS" "$NUM_TIME_SAMPLES" "$METRICS" "$LPIPS_NET"
done

# Aggregate results
echo ""
echo "=========================================="
echo "Aggregating results across all scenes..."
echo "=========================================="
python scripts/aggregate_sensitivity_results.py \
    --dataset "$DATASET" \
    --scenes "${SCENES[@]}"

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "Individual results: output/$DATASET/<scene>/sensitivity_analysis/"
echo "Aggregated results: output/$DATASET/aggregated_analysis/"
echo "=========================================="
