#!/bin/bash
# Static Gaussian Sensitivity Analysis Script
# 运行静止高斯球阈值敏感性分析

# 默认参数
SCENE=${1:-"coffee_martini"}
DATASET=${2:-"dynerf"}
NUM_LAMBDAS=${3:-25}
NUM_TIME_SAMPLES=${4:-50}

# 根据数据集选择配置文件
if [ "$DATASET" = "dynerf" ]; then
    CONFIG="arguments/dynerf/default.py"
    DATA_PATH="data/dynerf/${SCENE}"
    MODEL_PATH="output/dynerf/${SCENE}"
elif [ "$DATASET" = "dnerf" ]; then
    CONFIG="arguments/dnerf/default.py"
    DATA_PATH="data/dnerf/${SCENE}"
    MODEL_PATH="output/dnerf/${SCENE}"
elif [ "$DATASET" = "hypernerf" ]; then
    CONFIG="arguments/hypernerf/default.py"
    DATA_PATH="data/hypernerf/interp/${SCENE}"
    MODEL_PATH="output/hypernerf/${SCENE}"
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
echo "Num Lambdas: $NUM_LAMBDAS"
echo "Num Time Samples: $NUM_TIME_SAMPLES"
echo "=========================================="

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please train the model first."
    exit 1
fi

# 运行分析
python sensitivity_analysis.py \
    -m "$MODEL_PATH" \
    -s "$DATA_PATH" \
    --configs "$CONFIG" \
    --num_lambdas "$NUM_LAMBDAS" \
    --num_time_samples "$NUM_TIME_SAMPLES"

echo ""
echo "Analysis complete!"
echo "Results saved to: $MODEL_PATH/sensitivity_analysis/"
