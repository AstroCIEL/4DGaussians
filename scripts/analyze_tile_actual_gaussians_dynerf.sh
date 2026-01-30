#!/bin/bash
# DyNeRF Tile Actual Gaussians Analysis - 分析dynerf数据集中各场景test view的tile实际处理高斯球统计（考虑early stop）
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
echo "DyNeRF Tile Actual Gaussians Analysis (with Early Stop)"
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
echo ""
echo "注意: 此脚本需要重新编译CUDA扩展以支持n_contrib返回"
echo "请先运行以下命令重新编译:"
echo "  cd submodules/depth-diff-gaussian-rasterization"
echo "  pip install . --force-reinstall --no-deps"
echo "=========================================="

# 运行分析
python scripts/analyze_tile_actual_gaussians.py \
    --dataset "$DATASET" \
    --base_dir output \
    --iteration "$ITERATION" \
    --output_dir output \
    ${FRAME_IDX:+--frame_idx $FRAME_IDX}

echo ""
echo "=========================================="
echo "分析完成！"
echo "单个场景结果: output/$DATASET/<scene>/tile_actual_gaussians_analysis/"
echo "汇总结果: output/$DATASET/tile_actual_gaussians_analysis/aggregated_analysis/"
echo "=========================================="
