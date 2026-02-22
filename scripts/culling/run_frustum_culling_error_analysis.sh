#!/bin/bash
# Frustum Culling误差分析脚本
# 分析某数据集所有场景中：不经过deformation直接culling，相比正常流程造成的错误culling

# 获取脚本所在目录，并切换到项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT" || exit 1

# 默认参数
DATASET="dynerf"
BASE_DIR="output"
ITERATION=-1
OUTPUT_DIR="output"
CUDA_VISIBLE_DEVICES_VALUE=3

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
        --cuda_visible_devices)
            CUDA_VISIBLE_DEVICES_VALUE="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法:"
            echo "  $0 --dataset <dataset_name> [--base_dir output] [--iteration -1] [--output_dir output] [--cuda_visible_devices 0]"
            echo ""
            echo "示例:"
            echo "  $0 --dataset dynerf --iteration 30000 --cuda_visible_devices 3"
            echo "  $0 --dataset dnerf --base_dir output --output_dir output"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "错误: 请提供 --dataset"
    echo "使用 --help 查看帮助"
    exit 1
fi

if [ -n "$CUDA_VISIBLE_DEVICES_VALUE" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

echo "开始分析数据集: $DATASET"
python scripts/culling/analyze_frustum_culling_error.py \
    --dataset "$DATASET" \
    --base_dir "$BASE_DIR" \
    --iteration "$ITERATION" \
    --output_dir "$OUTPUT_DIR"
