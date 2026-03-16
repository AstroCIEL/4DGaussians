#!/usr/bin/env python3
"""
分析一个数据集下所有场景训练后高斯中心点的边界信息。

边界定义：
- 对每个场景，读取训练好的 point_cloud.ply 中所有高斯中心点 (x, y, z)
- 计算外接长方体：xyz_min / xyz_max
- 计算尺寸：xyz_size = xyz_max - xyz_min

说明：
- 仅打印到终端，不保存任何文件
"""

import os
import re
import sys
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple

import numpy as np
from plyfile import PlyData


def _find_scene_dirs(dataset_dir: str) -> List[str]:
    """查找数据集目录下包含 point_cloud 的场景目录。"""
    if not os.path.isdir(dataset_dir):
        return []

    scene_dirs: List[str] = []
    for item in sorted(os.listdir(dataset_dir)):
        scene_path = os.path.join(dataset_dir, item)
        if not os.path.isdir(scene_path):
            continue
        if os.path.isdir(os.path.join(scene_path, "point_cloud")):
            scene_dirs.append(scene_path)
    return scene_dirs


def _extract_iteration_num(folder_name: str) -> Optional[int]:
    """从 iteration_XXXXX 中提取迭代数。"""
    match = re.fullmatch(r"iteration_(\d+)", folder_name)
    if match is None:
        return None
    return int(match.group(1))


def _resolve_iteration_dir(point_cloud_root: str, iteration: int) -> Tuple[Optional[str], Optional[int]]:
    """
    根据指定 iteration 找到对应目录。
    - iteration = -1: 使用最大的 iteration_*
    - 其它值: 使用 iteration_<value>
    """
    if not os.path.isdir(point_cloud_root):
        return None, None

    if iteration >= 0:
        iter_dir = os.path.join(point_cloud_root, f"iteration_{iteration}")
        if os.path.isdir(iter_dir):
            return iter_dir, iteration
        return None, None

    candidates: List[Tuple[int, str]] = []
    for name in os.listdir(point_cloud_root):
        iter_num = _extract_iteration_num(name)
        if iter_num is None:
            continue
        full_path = os.path.join(point_cloud_root, name)
        if os.path.isdir(full_path):
            candidates.append((iter_num, full_path))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0])
    best_iter, best_dir = candidates[-1]
    return best_dir, best_iter


def _compute_bounds_from_ply(ply_path: str) -> Dict[str, np.ndarray]:
    """读取 point_cloud.ply 并计算 xyz 边界。"""
    ply_data = PlyData.read(ply_path)
    vertex = ply_data["vertex"]

    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float64),
            np.asarray(vertex["y"], dtype=np.float64),
            np.asarray(vertex["z"], dtype=np.float64),
        ],
        axis=1,
    )

    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    xyz_size = xyz_max - xyz_min
    xyz_center = 0.5 * (xyz_min + xyz_max)

    return {
        "count": np.array([xyz.shape[0]], dtype=np.int64),
        "xyz_min": xyz_min,
        "xyz_max": xyz_max,
        "xyz_size": xyz_size,
        "xyz_center": xyz_center,
    }


def _fmt_vec(vec: np.ndarray) -> str:
    return f"[{vec[0]: .6f}, {vec[1]: .6f}, {vec[2]: .6f}]"


def analyze_dataset(dataset: str, base_dir: str, iteration: int) -> int:
    dataset_dir = os.path.join(base_dir, dataset)
    if not os.path.isdir(dataset_dir):
        print(f"错误: 数据集目录不存在: {dataset_dir}")
        return 1

    scene_dirs = _find_scene_dirs(dataset_dir)
    if not scene_dirs:
        print(f"错误: 在 {dataset_dir} 下没有找到包含 point_cloud 的场景目录。")
        return 1

    print("=" * 96)
    print("Gaussian Bounding Box Analysis (Per Scene)")
    print("=" * 96)
    print(f"Dataset       : {dataset}")
    print(f"Dataset Dir   : {dataset_dir}")
    print(f"Iteration     : {'latest' if iteration < 0 else iteration}")
    print(f"Num Scenes    : {len(scene_dirs)}")
    print("=" * 96)

    scene_results = []
    for scene_path in scene_dirs:
        scene_name = os.path.basename(scene_path)
        pc_root = os.path.join(scene_path, "point_cloud")
        iter_dir, used_iter = _resolve_iteration_dir(pc_root, iteration)
        if iter_dir is None or used_iter is None:
            print(f"[跳过] {scene_name}: 未找到可用迭代目录")
            continue

        ply_path = os.path.join(iter_dir, "point_cloud.ply")
        if not os.path.isfile(ply_path):
            print(f"[跳过] {scene_name}: 文件不存在 {ply_path}")
            continue

        try:
            bounds = _compute_bounds_from_ply(ply_path)
            scene_results.append(
                {
                    "scene": scene_name,
                    "iteration": used_iter,
                    "count": int(bounds["count"][0]),
                    "xyz_min": bounds["xyz_min"],
                    "xyz_max": bounds["xyz_max"],
                    "xyz_size": bounds["xyz_size"],
                    "xyz_center": bounds["xyz_center"],
                }
            )
        except Exception as exc:
            print(f"[跳过] {scene_name}: 解析失败 ({exc})")

    if not scene_results:
        print("错误: 没有成功分析任何场景。")
        return 1

    scene_results.sort(key=lambda x: x["scene"])

    print("\n" + "-" * 96)
    print("Per-scene Bounds")
    print("-" * 96)
    for item in scene_results:
        print(f"Scene: {item['scene']}  (iteration={item['iteration']}, gaussians={item['count']})")
        print(f"  xyz_min   : {_fmt_vec(item['xyz_min'])}")
        print(f"  xyz_max   : {_fmt_vec(item['xyz_max'])}")
        print(f"  xyz_size  : {_fmt_vec(item['xyz_size'])}  # 外接长方体尺寸")
        print(f"  xyz_center: {_fmt_vec(item['xyz_center'])}")
        print("-" * 96)

    sizes = np.stack([x["xyz_size"] for x in scene_results], axis=0)
    volumes = sizes[:, 0] * sizes[:, 1] * sizes[:, 2]
    max_volume_idx = int(np.argmax(volumes))
    max_diag_idx = int(np.argmax(np.linalg.norm(sizes, axis=1)))

    all_min = np.min(np.stack([x["xyz_min"] for x in scene_results], axis=0), axis=0)
    all_max = np.max(np.stack([x["xyz_max"] for x in scene_results], axis=0), axis=0)
    all_size = all_max - all_min

    print("\n" + "=" * 96)
    print("Dataset-level Summary")
    print("=" * 96)
    print(f"Largest Volume Scene : {scene_results[max_volume_idx]['scene']}  (volume={volumes[max_volume_idx]:.6f})")
    print(
        "Largest Diag Scene   : "
        f"{scene_results[max_diag_idx]['scene']}  "
        f"(diag={np.linalg.norm(sizes[max_diag_idx]):.6f})"
    )
    print(f"Union xyz_min        : {_fmt_vec(all_min)}")
    print(f"Union xyz_max        : {_fmt_vec(all_max)}")
    print(f"Union xyz_size       : {_fmt_vec(all_size)}")
    print("=" * 96)

    return 0


def main() -> None:
    parser = ArgumentParser(description="Analyze Gaussian center bounds for all scenes in a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称，例如 dynerf / dnerf / hypernerf/interp")
    parser.add_argument("--base_dir", type=str, default="output", help="模型输出根目录")
    parser.add_argument("--iteration", type=int, default=-1, help="指定迭代号；-1 表示每个场景自动选最新迭代")
    args = parser.parse_args()

    exit_code = analyze_dataset(args.dataset, args.base_dir, args.iteration)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
