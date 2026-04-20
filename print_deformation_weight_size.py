#!/usr/bin/env python3
"""
根据数据集与 scene，打印训练好的 deformation 网络权重大小信息。

默认模型路径: output/<dataset>/<scene>
权重文件: <model_path>/point_cloud/iteration_<iter>/deformation.pth
"""

import os
import re
import sys
from argparse import ArgumentParser
from typing import Optional, Tuple


def _extract_iter_num(name: str, prefix: str) -> Optional[int]:
    pattern = rf"{re.escape(prefix)}(\d+)$"
    match = re.fullmatch(pattern, name)
    if match is None:
        return None
    return int(match.group(1))


def _resolve_iteration_dir(
    point_cloud_root: str, iteration: int, coarse: bool
) -> Tuple[Optional[str], Optional[int]]:
    if not os.path.isdir(point_cloud_root):
        return None, None

    prefix = "coarse_iteration_" if coarse else "iteration_"

    if iteration >= 0:
        iter_dir = os.path.join(point_cloud_root, f"{prefix}{iteration}")
        if os.path.isdir(iter_dir):
            return iter_dir, iteration
        return None, None

    candidates = []
    for name in os.listdir(point_cloud_root):
        iter_num = _extract_iter_num(name, prefix)
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


def _format_size_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024.0:.2f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024.0 * 1024.0):.2f} MB"
    return f"{num_bytes / (1024.0 * 1024.0 * 1024.0):.2f} GB"


def _format_param_line(name: str, weight_shape, bias_shape=None) -> str:
    if bias_shape is None:
        return f"{name}: weight{tuple(weight_shape)}"
    return f"{name}: weight{tuple(weight_shape)} + bias{tuple(bias_shape)}"


def _count_params(state, keys):
    total_params = 0
    total_bytes = 0
    for key in keys:
        value = state.get(key)
        if value is None or not hasattr(value, "numel"):
            continue
        numel = int(value.numel())
        total_params += numel
        try:
            total_bytes += int(numel * value.element_size())
        except Exception:
            pass
    return total_params, total_bytes


def _try_report_param_stats(weight_path: str) -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"提示: 未能导入 torch，跳过参数统计 ({exc})")
        return

    try:
        state = torch.load(weight_path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"提示: 读取权重失败，跳过参数统计 ({exc})")
        return

    if not isinstance(state, dict):
        print("提示: 权重文件不是 state_dict，跳过参数统计")
        return

    exclude_keys = {
        "time_poc",
        "pos_poc",
        "rotation_scaling_poc",
        "opacity_poc",
    }

    def _is_excluded(key: str) -> bool:
        if key in exclude_keys:
            return True
        if key.endswith(".aabb"):
            return True
        return False

    all_keys = [k for k in state.keys() if not _is_excluded(k)]
    grid_keys = [k for k in all_keys if k.startswith("deformation_net.grid.grids.")]
    mlp_keys = [k for k in all_keys if k not in grid_keys]

    total_params, total_bytes = _count_params(state, all_keys)
    grid_params, grid_bytes = _count_params(state, grid_keys)
    mlp_params, mlp_bytes = _count_params(state, mlp_keys)

    if total_params > 0:
        print(f"参数量(总计)   : {total_params}")
    if total_bytes > 0:
        print(f"参数占用(总计): {_format_size_bytes(total_bytes)}")
    if grid_params > 0:
        print(f"HexPlane参数量 : {grid_params}")
        print(f"HexPlane占用  : {_format_size_bytes(grid_bytes)}")
    if mlp_params > 0:
        print(f"MLP参数量     : {mlp_params}")
        print(f"MLP占用       : {_format_size_bytes(mlp_bytes)}")

    _try_report_structure(state)


def _try_report_structure(state) -> None:
    if not isinstance(state, dict):
        return

    print("\n网络结构概览")
    print("-" * 88)

    def _get_weight(key):
        value = state.get(key)
        if value is None or not hasattr(value, "shape"):
            return None
        return value

    # Time MLP
    time_w0 = _get_weight("timenet.0.weight")
    time_b0 = _get_weight("timenet.0.bias")
    time_w1 = _get_weight("timenet.2.weight")
    time_b1 = _get_weight("timenet.2.bias")
    if time_w0 is not None and time_w1 is not None:
        print("TimeNet:")
        print(f"  {_format_param_line('Linear0', time_w0.shape, time_b0.shape if time_b0 is not None else None)}")
        print(f"  ReLU")
        print(f"  {_format_param_line('Linear1', time_w1.shape, time_b1.shape if time_b1 is not None else None)}")
    else:
        print("TimeNet: 未在权重中找到对应参数")

    # Deformation feature_out
    feature_weights = []
    for key in state.keys():
        if key.startswith("deformation_net.feature_out") and key.endswith(".weight"):
            feature_weights.append((key, state[key]))
    feature_weights.sort(key=lambda x: x[0])
    if feature_weights:
        print("Deformation.feature_out:")
        for idx, (_, weight) in enumerate(feature_weights):
            in_dim = int(weight.shape[1])
            out_dim = int(weight.shape[0])
            if idx == 0:
                print(f"  Linear(in={in_dim}, out={out_dim})")
            else:
                print("  ReLU")
                print(f"  Linear(in={in_dim}, out={out_dim})")
    else:
        print("Deformation.feature_out: 未在权重中找到对应参数")

    # Heads
    heads = [
        ("pos_deform", 3),
        ("scales_deform", 3),
        ("rotations_deform", 4),
        ("opacity_deform", 1),
        ("shs_deform", 48),
        ("static_mlp", 1),
    ]
    for head, out_dim in heads:
        head_w0 = _get_weight(f"deformation_net.{head}.1.weight")
        head_b0 = _get_weight(f"deformation_net.{head}.1.bias")
        head_w1 = _get_weight(f"deformation_net.{head}.3.weight")
        head_b1 = _get_weight(f"deformation_net.{head}.3.bias")
        if head_w0 is None or head_w1 is None:
            continue
        print(f"{head}:")
        print(f"  ReLU")
        print(f"  {_format_param_line('Linear0', head_w0.shape, head_b0.shape if head_b0 is not None else None)}")
        print(f"  ReLU")
        print(f"  {_format_param_line('Linear1', head_w1.shape, head_b1.shape if head_b1 is not None else None)}")
        if head_w1.shape[0] != out_dim:
            print(f"  输出维度(权重推断): {int(head_w1.shape[0])}")

    # HexPlane grids
    grid_keys = [k for k in state.keys() if k.startswith("deformation_net.grid.grids.")]
    if grid_keys:
        grid_keys.sort()
        print("HexPlane grids:")
        for key in grid_keys:
            shape = tuple(state[key].shape)
            print(f"  {key}: {shape}")


def main() -> int:
    parser = ArgumentParser(description="打印 deformation 网络权重大小")
    parser.add_argument("--dataset", type=str, help="数据集名称，例如 dynerf / dnerf / hypernerf/interp")
    parser.add_argument("--scene", type=str, help="场景名称，例如 cut_roasted_beef / lego")
    parser.add_argument("--base_output", type=str, default="output", help="模型输出根目录")
    parser.add_argument("--model_path", type=str, default="", help="直接指定模型路径")
    parser.add_argument("--iteration", type=int, default=-1, help="指定迭代号；-1 表示自动选最新迭代")
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="使用 coarse_iteration_* 目录（默认使用 iteration_*）",
    )
    args = parser.parse_args()

    if args.model_path:
        model_path = args.model_path
    else:
        if not args.dataset or not args.scene:
            print("错误: 未提供 --model_path 时，必须提供 --dataset 与 --scene")
            return 1
        model_path = os.path.join(args.base_output, args.dataset, args.scene)

    if not os.path.isdir(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return 1

    pc_root = os.path.join(model_path, "point_cloud")
    iter_dir, used_iter = _resolve_iteration_dir(pc_root, args.iteration, args.coarse)
    if iter_dir is None or used_iter is None:
        kind = "coarse_iteration_*" if args.coarse else "iteration_*"
        print(f"错误: 未找到可用迭代目录（{kind}）: {pc_root}")
        return 1

    weight_path = os.path.join(iter_dir, "deformation.pth")
    if not os.path.isfile(weight_path):
        print(f"错误: 权重文件不存在: {weight_path}")
        return 1

    file_size = os.path.getsize(weight_path)

    print("=" * 88)
    print("Deformation Weight Size")
    print("=" * 88)
    print(f"Model Path     : {model_path}")
    print(f"Iteration      : {used_iter}")
    print(f"Iteration Dir  : {iter_dir}")
    print(f"Weight File    : {weight_path}")
    print(f"文件大小       : {_format_size_bytes(file_size)}")
    print("-" * 88)
    _try_report_param_stats(weight_path)
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
