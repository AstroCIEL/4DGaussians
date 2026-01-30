#!/bin/bash
# DyNeRF Tile Gaussians Analysis - 分析dynerf数据集中各场景test view的tile高斯球统计
export CUDA_VISIBLE_DEVICES=3

SCENES=(
    "cut_roasted_beef"
    "coffee_martini"
    "cook_spinach"
    "flame_salmon_1"
    "flame_steak"
    "sear_steak"
)

DATASET="dynerf"
ITERATION=14000
FRAME_IDX=${1:-}  # 可选：指定要分析的帧索引，如果不指定则分析所有test view

echo "=========================================="
echo "DyNeRF Tile Gaussians Analysis"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Iteration: $ITERATION"
if [ -n "$FRAME_IDX" ]; then
    echo "Frame Index: $FRAME_IDX (分析单个帧)"
else
    echo "Frame Index: 全部 (分析所有test view)"
fi
echo "Scenes: ${SCENES[@]}"
echo "=========================================="

# 运行分析
python scripts/analyze_tile_gaussians.py \
    --dataset "$DATASET" \
    --base_dir output \
    --iteration "$ITERATION" \
    --output_dir output \
    ${FRAME_IDX:+--frame_idx $FRAME_IDX}

echo ""
echo "=========================================="
echo "分析完成！"
echo "单个场景结果: output/$DATASET/<scene>/tile_gaussians_analysis/"
echo "汇总结果: output/$DATASET/tile_gaussians_analysis/aggregated_analysis/"
echo "=========================================="
