#!/bin/bash

workdir=$1
datatype=$2 # blender, hypernerf, llff

# -------------------------------------------------------
# 1. 环境设置
# -------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0
# 尝试禁用 Qt 图形界面依赖
export QT_QPA_PLATFORM=offscreen 

# -------------------------------------------------------
# 2. 清理旧数据 (防止上次崩溃残留导致错误)
# -------------------------------------------------------
echo "Cleaning up old files..."
rm -rf $workdir/sparse_
rm -rf $workdir/image_colmap
rm -rf $workdir/colmap

# -------------------------------------------------------
# 3. 数据转换与准备
# -------------------------------------------------------
echo "Running data conversion..."
python scripts/"$datatype"2colmap.py $workdir

# 删除转换脚本可能生成的临时目录
rm -rf $workdir/colmap/sparse/0

# 创建目录结构
mkdir -p $workdir/colmap/images
mkdir -p $workdir/colmap/sparse/0
mkdir -p $workdir/colmap/dense/workspace

# 复制图片
# 注意：确保这里复制源正确。如果 $workdir/image_colmap 不存在，可能在 python 脚本步骤已生成在别处
if [ -d "$workdir/image_colmap" ]; then
    cp -r $workdir/image_colmap/* $workdir/colmap/images/
else
    echo "Warning: $workdir/image_colmap not found, checking if images are already in place."
fi

# 复制 sparse_custom (如果存在)
if [ -d "$workdir/sparse_" ]; then
    cp -r $workdir/sparse_ $workdir/colmap/sparse_custom
fi

# -------------------------------------------------------
# 4. 特征提取 (CPU Mode)
# -------------------------------------------------------
echo "--- [Step 1] Feature Extraction (CPU) ---"
# 添加 --SiftExtraction.use_gpu 0 确保稳定
colmap feature_extractor \
    --database_path $workdir/colmap/database.db \
    --image_path $workdir/colmap/images \
    --SiftExtraction.max_image_size 4096 \
    --SiftExtraction.max_num_features 16384 \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1 \
    --SiftExtraction.use_gpu 0

echo "--- Database Setup ---"
# 只有当 cameras.txt 存在时才运行 database.py
if [ -f "$workdir/colmap/sparse_custom/cameras.txt" ]; then
    python database.py \
        --database_path $workdir/colmap/database.db \
        --txt_path $workdir/colmap/sparse_custom/cameras.txt
fi

# -------------------------------------------------------
# 5. 特征匹配 (CPU Mode) - 关键修复点！
# -------------------------------------------------------
echo "--- [Step 2] Exhaustive Matcher (CPU) ---"
# 必须加上 --SiftMatching.use_gpu 0 否则会在无头服务器上崩溃
colmap exhaustive_matcher \
    --database_path $workdir/colmap/database.db \
    --SiftMatching.use_gpu 0

# -------------------------------------------------------
# 6. 三角测量 (Sparse Reconstruction)
# -------------------------------------------------------
echo "--- [Step 3] Point Triangulator ---"
colmap point_triangulator \
    --database_path $workdir/colmap/database.db \
    --image_path $workdir/colmap/images \
    --input_path $workdir/colmap/sparse_custom \
    --output_path $workdir/colmap/sparse/0 \
    --clear_points 1

# -------------------------------------------------------
# 7. 稠密重建 (尝试 CUDA)
# -------------------------------------------------------
echo "--- [Step 4] Image Undistortion ---"
colmap image_undistorter \
    --image_path $workdir/colmap/images \
    --input_path $workdir/colmap/sparse/0 \
    --output_path $workdir/colmap/dense/workspace

echo "--- [Step 5] Stereo Reconstruction (Try CUDA) ---"
# 尝试运行 Patch Match
colmap patch_match_stereo \
    --workspace_path $workdir/colmap/dense/workspace

# 检查 patch_match 是否成功
if [ $? -eq 0 ]; then
    echo "--- [Step 6] Stereo Fusion ---"
    colmap stereo_fusion \
        --workspace_path $workdir/colmap/dense/workspace \
        --output_path $workdir/colmap/dense/workspace/fused.ply
else
    echo "CUDA Stereo Reconstruction failed or skipped."
    echo "Will fallback to sparse point cloud."
fi

# -------------------------------------------------------
# 8. 结果检查与回退处理 (修复版)
# -------------------------------------------------------
FUSED_PLY="$workdir/colmap/dense/workspace/fused.ply"
SPARSE_DIR="$workdir/colmap/sparse/0"
UNDISTORTED_DIR="$workdir/colmap/dense/workspace/sparse" # image_undistorter 输出的稀疏重建位置

# 1. 检查是否已经生成了 fused.ply (CUDA 成功的情况)
if [ -f "$FUSED_PLY" ]; then
    echo "Success: Dense point cloud generated at $FUSED_PLY"
    exit 0
fi

echo "Warning: Dense reconstruction failed. Attempting to convert Sparse Point Cloud..."

# 2. 寻找稀疏点云文件 (可能是 .ply 或 .bin)
# 优先找 image_undistorter 输出的目录 (通常是 dense/workspace/sparse)
if [ -f "$UNDISTORTED_DIR/points3D.ply" ]; then
    SOURCE_PLY="$UNDISTORTED_DIR/points3D.ply"
elif [ -f "$UNDISTORTED_DIR/points3D.bin" ]; then
    SOURCE_BIN="$UNDISTORTED_DIR/points3D.bin"
# 其次找原始 sparse 目录
elif [ -f "$SPARSE_DIR/points3D.ply" ]; then
    SOURCE_PLY="$SPARSE_DIR/points3D.ply"
elif [ -f "$SPARSE_DIR/points3D.bin" ]; then
    SOURCE_BIN="$SPARSE_DIR/points3D.bin"
fi

# 3. 执行转换或复制
if [ -n "$SOURCE_PLY" ]; then
    echo "Found sparse PLY at $SOURCE_PLY. Copying to fused.ply..."
    cp "$SOURCE_PLY" "$FUSED_PLY"
elif [ -n "$SOURCE_BIN" ]; then
    echo "Found sparse BIN at $SOURCE_BIN. Converting to fused.ply..."
    # 使用 colmap model_converter 将 bin 转为 ply
    colmap model_converter \
        --input_path $(dirname "$SOURCE_BIN") \
        --output_path "$FUSED_PLY" \
        --output_type PLY
else
    echo "Error: Reconstruction failed completely. No sparse point cloud found in:"
    echo "  - $UNDISTORTED_DIR"
    echo "  - $SPARSE_DIR"
    # 列出目录内容帮助调试
    echo "Listing $workdir/colmap recursively:"
    ls -R "$workdir/colmap"
    exit 1
fi

# 4. 最终检查
if [ -f "$FUSED_PLY" ]; then
    echo "Fallback Success: Created $FUSED_PLY from sparse reconstruction."
else
    echo "Error: Failed to create fused.ply even after fallback attempt."
    exit 1
fi

echo "Done."