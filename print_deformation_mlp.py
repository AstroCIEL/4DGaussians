#!/usr/bin/env python3
"""
加载已训练场景的高斯模型，并打印形变场（deformation network）中 MLP 的结构。
用法示例:
  python print_deformation_mlp.py -m output/dynerf/coffee_martini --iteration 14000
  python print_deformation_mlp.py -m output/dynerf/coffee_martini --configs configs/xxx.py
"""
import os
import sys
import torch.nn as nn
from argparse import ArgumentParser

from arguments import ModelParams, ModelHiddenParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from utils.general_utils import safe_state


def print_linear(module, name, indent=0):
    """打印单个 Linear 层信息。"""
    if isinstance(module, nn.Linear):
        print(" " * indent + f"  Linear: in_features={module.in_features}, out_features={module.out_features}")
        return True
    return False


def print_mlp_structure(module, name_prefix="", indent=0):
    """递归打印 MLP/Sequential 结构，只展开到 Linear 层。"""
    if isinstance(module, nn.Linear):
        print(" " * indent + f"{name_prefix}: Linear(in={module.in_features}, out={module.out_features})")
        return
    if isinstance(module, (nn.Sequential, nn.ModuleList)):
        for i, child in enumerate(module):
            sub_name = f"{name_prefix}[{i}]" if name_prefix else f"[{i}]"
            if isinstance(child, nn.Linear):
                print(" " * indent + f"{sub_name}: Linear(in={child.in_features}, out={child.out_features})")
            elif isinstance(child, (nn.ReLU, nn.Sigmoid)):
                print(" " * indent + f"{sub_name}: {child.__class__.__name__}")
            else:
                print_mlp_structure(child, sub_name, indent)
    else:
        for sub_name, child in module.named_children():
            full_name = f"{name_prefix}.{sub_name}" if name_prefix else sub_name
            print_mlp_structure(child, full_name, indent)


def print_deformation_mlp(gaussians):
    """打印形变场中 MLP 的完整结构。"""
    deform = gaussians._deformation
    net = deform.deformation_net

    print("\n" + "=" * 60)
    print("形变场网络 (deform_network) 概览")
    print("=" * 60)
    print(f"  MLP 宽度 W (net_width): {net.W}")
    print(f"  MLP 深度 D (defor_depth): {net.D}")
    print(f"  grid_pe: {net.grid_pe}, no_grid: {net.no_grid}")
    if not net.no_grid:
        print(f"  grid feature_dim (输入到 MLP 的维度): {net.grid.feat_dim}")
    print()

    print("-" * 60)
    print("1. 主干: feature_out (grid_feat -> 共享隐层)")
    print("-" * 60)
    print_mlp_structure(net.feature_out, "feature_out", indent=0)
    print()

    print("-" * 60)
    print("2. 各输出头 (共享隐层 -> 形变输出)")
    print("-" * 60)
    for head_name in ["pos_deform", "scales_deform", "rotations_deform", "opacity_deform", "shs_deform"]:
        if hasattr(net, head_name):
            head = getattr(net, head_name)
            print(f"  {head_name}:")
            print_mlp_structure(head, "", indent=4)
            print()
    print("=" * 60)


def main():
    parser = ArgumentParser(description="Load scene Gaussians and print deformation MLP structure.")
    model = ModelParams(parser, sentinel=True)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", type=int, default=-1, help="checkpoint iteration, -1 = latest")
    parser.add_argument("--configs", type=str, default="")
    args = get_combined_args(parser)

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    # 用 checkpoint 目录下保存的 cfg_args 覆盖形变场相关超参，避免 parser 默认值覆盖导致 load_state_dict 结构不匹配
    cfg_path = os.path.join(args.model_path, "cfg_args")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r") as f:
            saved_cfg = eval(f.read())
        for key in (
            "net_width", "defor_depth", "kplanes_config", "multires",
            "no_grid", "grid_pe", "bounds",
        ):
            if hasattr(saved_cfg, key):
                setattr(args, key, getattr(saved_cfg, key))

    safe_state(getattr(args, "quiet", False))
    dataset = model.extract(args)
    hp = hyperparam.extract(args)

    print(f"Loading scene: model_path={dataset.model_path}, iteration={args.iteration}")
    gaussians = GaussianModel(dataset.sh_degree, hp)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    print(f"Loaded iteration: {scene.loaded_iter}, #points: {gaussians.get_xyz.shape[0]}")
    print_deformation_mlp(gaussians)


if __name__ == "__main__":
    main()
