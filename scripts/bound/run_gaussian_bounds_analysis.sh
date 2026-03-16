#!/bin/bash
# 高斯中心点边界分析脚本（数据集级）
# 统计数据集中每个场景训练后高斯中心点集合的外接长方体尺寸，并打印到终端

set -e

# 获取脚本所在目录，并切换到项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT" || exit 1

# 默认参数
DATASET="dynerf"
BASE_DIR="output"
ITERATION=-1

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
        -h|--help)
            echo "用法:"
            echo "  $0 --dataset <dataset_name> [--base_dir output] [--iteration -1]"
            echo ""
            echo "示例:"
            echo "  $0 --dataset dynerf"
            echo "  $0 --dataset dnerf --iteration 30000"
            echo "  $0 --dataset hypernerf/interp --base_dir output"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Gaussian Bounds Analysis"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Base Dir: $BASE_DIR"
echo "Iteration: $ITERATION"
echo "=========================================="

python scripts/bound/analyze_gaussian_bounds.py \
    --dataset "$DATASET" \
    --base_dir "$BASE_DIR" \
    --iteration "$ITERATION"
