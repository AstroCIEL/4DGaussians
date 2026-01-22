#!/bin/bash
# Static Gaussian Sensitivity Analysis Script
# 运行静止高斯球阈值敏感性分析
export CUDA_VISIBLE_DEVICES=3

# 默认参数
SCENE=${1:-"coffee_martini"}
DATASET=${2:-"dynerf"}
MODE=${3:-"ratio"}           # ratio | lambda
RATIO_STEP=${4:-5}
NUM_LAMBDAS=${5:-20}
NUM_TIME_SAMPLES=${6:-60}
METRICS=${7:-"psnr,ssim,lpips"}
LPIPS_NET=${8:-"vgg"}

# 根据数据集选择配置文件
if [ "$DATASET" = "dynerf" ]; then
    CONFIG="arguments/dynerf/${SCENE}.py"
    DATA_PATH="data/dynerf/${SCENE}"
    MODEL_PATH="output/dynerf/${SCENE}"
elif [ "$DATASET" = "dnerf" ]; then
    CONFIG="arguments/dnerf/${SCENE}.py"
    DATA_PATH="data/dnerf/${SCENE}"
    MODEL_PATH="output/dnerf/${SCENE}"
elif [ "$DATASET" = "hypernerf/interp" ]; then
    CONFIG="arguments/hypernerf/default.py"
    DATA_PATH="data/hypernerf/interp/${SCENE}"
    MODEL_PATH="output/hypernerf/interp/${SCENE}"
elif [ "$DATASET" = "hypernerf/vrig" ]; then
    CONFIG="arguments/hypernerf/${SCENE}.py"
    DATA_PATH="data/hypernerf/vrig/${SCENE}"
    MODEL_PATH="output/hypernerf/vrig/${SCENE}"
else
    echo "Unknown dataset: $DATASET"
    echo "Supported: dynerf, dnerf, hypernerf"
    exit 1
fi

echo "=========================================="
echo "Static Gaussian Sensitivity Analysis"
echo "=========================================="
echo "Scene: $SCENE"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "Model Path: $MODEL_PATH"
echo "Mode: $MODE"
echo "Static Ratio Step: ${RATIO_STEP}%"
echo "Num Lambdas: $NUM_LAMBDAS"
echo "Num Time Samples: $NUM_TIME_SAMPLES"
echo "Metrics: $METRICS"
echo "LPIPS Net: $LPIPS_NET"
echo "=========================================="

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please train the model first."
    exit 1
fi

# 运行分析
if [ "$MODE" = "ratio" ]; then
    python sensitivity_analysis.py \
        -m "$MODEL_PATH" \
        -s "$DATA_PATH" \
        --configs "$CONFIG" \
        --static_ratio_step "$RATIO_STEP" \
        --num_time_samples "$NUM_TIME_SAMPLES" \
        --metrics "$METRICS" \
        --lpips_net "$LPIPS_NET"
elif [ "$MODE" = "lambda" ]; then
    python sensitivity_analysis.py \
        -m "$MODEL_PATH" \
        -s "$DATA_PATH" \
        --configs "$CONFIG" \
        --num_lambdas "$NUM_LAMBDAS" \
        --num_time_samples "$NUM_TIME_SAMPLES" \
        --metrics "$METRICS" \
        --lpips_net "$LPIPS_NET"
else
    echo "Unknown mode: $MODE"
    echo "Supported: ratio, lambda"
    exit 1
fi

echo ""
echo "Analysis complete!"
echo "Results saved to: $MODEL_PATH/sensitivity_analysis/"
