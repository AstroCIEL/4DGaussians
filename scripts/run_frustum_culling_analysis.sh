#!/bin/bash
# Frustum Culling分析脚本
# 用于分析frustum culling统计信息
export CUDA_VISIBLE_DEVICES=3

# 获取脚本所在目录，并切换到项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT" || exit 1

# 默认参数
DATASET=""
BASE_DIR="output"
ITERATION=-1
OUTPUT_DIR="output"
MODEL_PATH=""
SOURCE_PATH=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --base_dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --iteration)
            ITERATION="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --source_path)
            SOURCE_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法:"
            echo "  分析整个数据集:"
            echo "    $0 --dataset <dataset_name> [--base_dir output] [--iteration 30000] [--output_dir output]"
            echo ""
            echo "  分析单个场景:"
            echo "    $0 --model_path <model_path> --source_path <source_path> [--iteration 30000] [--output_dir output]"
            echo ""
            echo "示例:"
            echo "  $0 --dataset dynerf --iteration 30000"
            echo "  $0 --model_path output/dynerf/bouncingballs --source_path data/dynerf/bouncingballs --iteration 30000"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查参数
if [ -z "$DATASET" ] && [ -z "$MODEL_PATH" ]; then
    echo "错误: 请提供 --dataset 或 --model_path"
    echo "使用 --help 查看帮助"
    exit 1
fi

if [ -n "$MODEL_PATH" ] && [ -z "$SOURCE_PATH" ]; then
    echo "错误: 使用 --model_path 时必须提供 --source_path"
    exit 1
fi

# 运行分析脚本
if [ -n "$DATASET" ]; then
    echo "分析数据集: $DATASET"
    python scripts/analyze_frustum_culling.py \
        --dataset "$DATASET" \
        --base_dir "$BASE_DIR" \
        --iteration "$ITERATION" \
        --output_dir "$OUTPUT_DIR"
else
    echo "分析场景: $MODEL_PATH"
    python scripts/analyze_frustum_culling.py \
        --model_path "$MODEL_PATH" \
        --source_path "$SOURCE_PATH" \
        --iteration "$ITERATION" \
        --output_dir "$OUTPUT_DIR"
fi
