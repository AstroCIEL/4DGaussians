#!/bin/bash

# 遇到错误立即停止执行，防止后续步骤基于错误的数据运行
set -e

# 数据集根目录
DATA_ROOT="data/dynerf"

#定义所有场景列表
SCENES=(
    "flame_salmon_1"
    "coffee_martini"
    "cook_spinach"
    "cut_roasted_beef"
    "flame_steak"
    "sear_steak"
)

echo "开始批量预处理..."

for scene in "${SCENES[@]}"; do
    scene_dir="$DATA_ROOT/$scene"
    
    echo "=================================================="
    echo "正在处理场景: $scene"
    echo "目录: $scene_dir"
    echo "=================================================="

    # ----------------------------------------
    # Step 1: 提取视频帧 (Extract frames)
    # ----------------------------------------
    echo "[1/3] 正在提取视频帧..."
    python scripts/preprocess_dynerf.py --datadir "$scene_dir"

    # ----------------------------------------
    # Step 2: 生成点云 (COLMAP)
    # ----------------------------------------
    echo "[2/3] 正在运行 COLMAP (这可能需要较长时间)..."
    # 注意：确保 colmap.sh 在当前路径下，或者使用绝对路径
    bash colmap.sh "$scene_dir" llff

    # ----------------------------------------
    # Step 3: 下采样点云 (Downsample)
    # ----------------------------------------
    echo "[3/3] 正在下采样点云..."
    
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