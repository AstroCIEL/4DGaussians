#!/bin/bash

# 遇到错误立即停止执行，防止后续步骤基于错误的数据运行
set -e

# 数据集根目录
DATA_ROOT="data/hypernerf"

#定义所有场景列表
SCENES=(
    "interp/aleks-teapot"
    "interp/chickchicken"
    "interp/cut-lemon1"
    "interp/slice-banana"
    "interp/torchocolate"
    "interp/hand1-dense-v2"
    "vrig/broom2"
    "vrig/vrig-3dprinter"
    "vrig/vrig-chicken"
    "vrig/vrig-peel-banana"
    "misc/americano"
    "misc/cross-hands1"
    "misc/espresso"
    "misc/keyboard"
    "misc/oven-mitts"
    "misc/split-cookie"
    "misc/tamping"
)

echo "开始批量预处理..."

for scene in "${SCENES[@]}"; do
    scene_dir="$DATA_ROOT/$scene"
    
    echo "=================================================="
    echo "正在处理场景: $scene"
    echo "目录: $scene_dir"
    echo "=================================================="
    # ----------------------------------------
    # Step 1: 生成点云 (COLMAP)
    # ----------------------------------------
    echo "[1/2] 正在运行 COLMAP..."
    # 注意：确保 colmap.sh 在当前路径下，或者使用绝对路径
    bash colmap.sh "$scene_dir" hypernerf

    # -----------------------------------------
    # Step 2: 下采样点云 (Downsample)
    # -----------------------------------------
    echo "[2/2] 正在下采样点云..."
    
    input_ply="$scene_dir/colmap/dense/workspace/fused.ply"
    output_ply="$scene_dir/points3D_downsample2.ply"

    # 检查 COLMAP 是否成功生成了 dense 结果
    if [ -f "$input_ply" ]; then
        python scripts/downsample_point.py "$input_ply" "$output_ply"
        echo "场景 $scene 处理完成！"
    else
        echo "错误: 未找到文件 $input_ply"
        echo "COLMAP 可能执行失败，请检查日志。"
        exit 1
    fi
    
    echo "" # 打印空行分隔
done

echo "所有场景预处理全部完成。"