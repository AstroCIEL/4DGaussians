"""
离线动静标签生成器：
- 读取 simulator/config 中指定的 dataset/scene 对应模型
- 使用 StaticGaussianAnalyzer 计算每个高斯球的最大形变量
- 按形变大小排序，根据 config 中的 static/quasi 比例打标签：0=静止，1=微动，2=巨变
- 将标签与统计信息写回模型目录，供 simulator 直接读取
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
from argparse import ArgumentParser, Namespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from arguments import ModelParams, PipelineParams, ModelHiddenParams
from scene import Scene
from gaussian_renderer import GaussianModel
from scene.static_analyzer import StaticGaussianAnalyzer
from utils.general_utils import safe_state


def _resolve_paths(config: dict):
    sim_cfg = config.get("simulation", {})
    dataset = sim_cfg.get("dataset")
    scene = sim_cfg.get("scene")
    model_path = sim_cfg.get("model_path") or config.get("model_path")
    if not model_path and dataset and scene:
        base = sim_cfg.get("base_output", "output")
        model_path = os.path.join(base, dataset, scene)
    source_path = sim_cfg.get("source_path") or config.get("source_path")
    if not source_path and dataset and scene:
        candidate = os.path.join("data", dataset, scene)
        if os.path.isdir(candidate):
            source_path = os.path.abspath(candidate)
    return model_path, source_path, dataset, scene


def _build_args(model_path: str, source_path: str, sim_cfg: dict):
    parser = ArgumentParser()
    model_param_group = ModelParams(parser, sentinel=True)
    pipeline_param_group = PipelineParams(parser)
    hidden_param_group = ModelHiddenParams(parser)

    args = Namespace()
    args.model_path = model_path
    args.source_path = source_path
    args.images = "images"
    args.resolution = -1
    args.white_background = True
    args.data_device = "cuda"
    args.eval = True
    args.render_process = False
    args.add_points = False
    args.extension = ".png"
    args.llffhold = 8
    args.sh_degree = 3

    cfg_file = os.path.join(model_path, "cfg_args")
    if os.path.isfile(cfg_file):
        try:
            with open(cfg_file, "r") as f:
                cfg_str = f.read()
            cfg = eval(cfg_str)
            for k, v in vars(cfg).items():
                if k not in ("model_path", "source_path"):
                    setattr(args, k, v)
        except Exception:
            pass

    model_params = model_param_group.extract(args)
    pipeline_params = pipeline_param_group.extract(args)
    hyperparam = hidden_param_group.extract(args)
    iteration = sim_cfg.get("iteration", -1)
    return model_params, pipeline_params, hyperparam, iteration


def generate_motion_labels_from_config(config: dict):
    sim_cfg = config.get("simulation", {})
    workload_cfg = config.get("workload", {})
    label_cfg = config.get("labeling", {})

    model_path, source_path, dataset, scene = _resolve_paths(config)
    if not model_path or not os.path.isdir(model_path):
        raise FileNotFoundError(f"model_path 不存在: {model_path}")
    if not source_path or not os.path.isdir(source_path):
        raise FileNotFoundError(f"source_path 不存在: {source_path}")

    static_ratio = workload_cfg.get("static_ratio", 0.4)
    quasi_ratio = workload_cfg.get("quasi_ratio", 0.4)
    dyn_ratio = max(0.0, 1.0 - static_ratio - quasi_ratio)
    num_time_samples = int(label_cfg.get("num_time_samples", 50))
    out_npy = label_cfg.get("output_npy", "motion_labels.npy")
    out_json = label_cfg.get("output_json", "motion_labels.json")

    model_params, pipeline_params, hyperparam, iteration = _build_args(model_path, source_path, sim_cfg)

    safe_state(False)
    gaussians = GaussianModel(model_params.sh_degree, hyperparam)
    scene_obj = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)

    print(f"[label] loaded model: {dataset}/{scene}, gaussians={gaussians._xyz.shape[0]}, iteration={iteration}")
    analyzer = StaticGaussianAnalyzer(gaussians, num_time_samples=num_time_samples)
    max_def = analyzer.compute_max_deformation(verbose=True).detach().cpu().numpy()

    n = len(max_def)
    static_n = int(n * static_ratio)
    quasi_n = int(n * quasi_ratio)
    dyn_n = n - static_n - quasi_n
    order = np.argsort(max_def)  # 从小到大：静止在前，巨变在后
    labels = np.zeros(n, dtype=np.int32)
    labels[order[static_n:static_n + quasi_n]] = 1
    labels[order[static_n + quasi_n:]] = 2

    out_dir = model_path
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, out_npy), labels)
    summary = {
        "counts": {"static": int(static_n), "quasi": int(quasi_n), "dynamic": int(dyn_n)},
        "ratios": {"static": static_ratio, "quasi": quasi_ratio, "dynamic": dyn_ratio},
        "thresholds": {
            "static_max": float(max_def[order[static_n - 1]]) if static_n > 0 else 0.0,
            "quasi_max": float(max_def[order[static_n + quasi_n - 1]]) if quasi_n > 0 else 0.0,
        },
        "num_time_samples": num_time_samples,
        "label_file": out_npy,
    }
    with open(os.path.join(out_dir, out_json), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[label] done. saved labels to {os.path.join(out_dir, out_npy)}, summary to {os.path.join(out_dir, out_json)}")
    return labels, summary


def generate_motion_labels(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return generate_motion_labels_from_config(config)


def parse_args():
    parser = ArgumentParser(description="Generate motion labels for a trained 4DGS model.")
    parser.add_argument("--config", type=str, default="simulator/configs/default.yaml", help="simulator config path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_motion_labels(args.config)
