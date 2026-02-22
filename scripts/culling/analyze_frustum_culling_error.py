#!/usr/bin/env python3
"""
分析Frustum Culling误差脚本

目标：
对某数据集所有场景，在video split相机上逐视角比较两条流程：
1) coarse：不经过deformation，直接frustum culling
2) fine：正常流程（先deformation，再frustum culling）

并统计：
- 错误被cull（false_cull）：coarse判为不可见，但fine判为可见
- 错误没被cull（false_keep）：coarse判为可见，但fine判为不可见
"""

import csv
import json
import os
import re
import sys
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录到Python路径，确保可以导入模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from arguments import ModelHiddenParams, ModelParams, PipelineParams
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state


def _load_cfg_namespace(model_path: str):
    """尽量解析模型目录中的cfg_args，返回Namespace或None。"""
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        return None

    try:
        with open(cfg_path, "r") as f:
            cfg_text = f.read()
        # cfg_args通常是字符串形式的 Namespace(...)
        cfg_ns = eval(cfg_text, {"Namespace": Namespace})
        return cfg_ns
    except Exception as exc:
        print(f"警告: 解析cfg_args失败 ({cfg_path}): {exc}")
        return None


def _resolve_source_path(model_path: str, dataset_name: str, scene_name: str):
    """优先从cfg_args读取source_path，失败时退化到data/<dataset>/<scene>。"""
    cfg_ns = _load_cfg_namespace(model_path)
    if cfg_ns is not None and hasattr(cfg_ns, "source_path"):
        source_path = getattr(cfg_ns, "source_path")
        if source_path:
            if os.path.isabs(source_path):
                return source_path
            return os.path.normpath(os.path.join(PROJECT_ROOT, source_path))

    # 兼容非标准cfg格式，尝试正则提取
    cfg_path = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg_text = f.read()
            match = re.search(r"source_path['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", cfg_text)
            if match:
                source_path = match.group(1)
                if os.path.isabs(source_path):
                    return source_path
                return os.path.normpath(os.path.join(PROJECT_ROOT, source_path))
        except Exception:
            pass

    fallback = os.path.join(PROJECT_ROOT, "data", dataset_name, scene_name)
    if os.path.exists(fallback):
        return fallback
    return None


def _build_scene_runtime_args(model_path: str, source_path: str):
    """
    构建Scene所需参数：先给默认值，再用cfg_args覆盖（但保留model/source路径）。
    """
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

    cfg_ns = _load_cfg_namespace(model_path)
    if cfg_ns is not None:
        for key, value in vars(cfg_ns).items():
            if key not in ("model_path", "source_path"):
                setattr(args, key, value)

    return args


def analyze_scene_frustum_culling_error(model_path: str, source_path: str, iteration: int):
    """分析单场景：比较coarse/fine可见性，统计误判。"""
    print(f"\n分析场景: {model_path}")
    print(f"source_path: {source_path}")

    parser = ArgumentParser()
    model_params_class = ModelParams(parser, sentinel=True)
    pipeline_params_class = PipelineParams(parser)
    hyperparam_class = ModelHiddenParams(parser)

    runtime_args = _build_scene_runtime_args(model_path, source_path)
    model_params = model_params_class.extract(runtime_args)
    pipeline_params = pipeline_params_class.extract(runtime_args)
    hyperparam = hyperparam_class.extract(runtime_args)

    gaussians = GaussianModel(model_params.sh_degree, hyperparam)
    scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)
    video_cameras = scene.getVideoCameras()

    total_gaussians = int(gaussians.get_xyz.shape[0])
    num_video_views = len(video_cameras)
    print(f"video视图数量: {num_video_views}")
    print(f"总高斯球数量: {total_gaussians}")

    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    per_view_stats = []

    for view_idx, view in enumerate(tqdm(video_cameras, desc="比较coarse/fine culling")):
        with torch.no_grad():
            coarse_results = render(
                view,
                gaussians,
                pipeline_params,
                background,
                stage="coarse",
                cam_type=scene.dataset_type,
            )
            fine_results = render(
                view,
                gaussians,
                pipeline_params,
                background,
                stage="fine",
                cam_type=scene.dataset_type,
            )

        coarse_vis = coarse_results["visibility_filter"]
        fine_vis = fine_results["visibility_filter"]

        if coarse_vis.shape != fine_vis.shape:
            raise RuntimeError(
                f"可见性张量尺寸不一致: coarse={coarse_vis.shape}, fine={fine_vis.shape}"
            )

        false_cull_mask = (~coarse_vis) & fine_vis
        false_keep_mask = coarse_vis & (~fine_vis)

        false_cull_count = int(false_cull_mask.sum().item())
        false_keep_count = int(false_keep_mask.sum().item())
        coarse_visible_count = int(coarse_vis.sum().item())
        fine_visible_count = int(fine_vis.sum().item())
        mismatch_count = false_cull_count + false_keep_count

        per_view_stats.append(
            {
                "view_idx": view_idx,
                "total_gaussians": total_gaussians,
                "coarse_visible_count": coarse_visible_count,
                "fine_visible_count": fine_visible_count,
                "false_cull_count": false_cull_count,
                "false_keep_count": false_keep_count,
                "mismatch_count": mismatch_count,
                "false_cull_ratio": (false_cull_count / total_gaussians) if total_gaussians > 0 else 0.0,
                "false_keep_ratio": (false_keep_count / total_gaussians) if total_gaussians > 0 else 0.0,
                "mismatch_ratio": (mismatch_count / total_gaussians) if total_gaussians > 0 else 0.0,
            }
        )

    if not per_view_stats:
        raise RuntimeError("场景没有可用的video视图。")

    false_cull_list = np.array([x["false_cull_count"] for x in per_view_stats], dtype=np.float64)
    false_keep_list = np.array([x["false_keep_count"] for x in per_view_stats], dtype=np.float64)
    mismatch_list = np.array([x["mismatch_count"] for x in per_view_stats], dtype=np.float64)
    false_cull_ratio_list = np.array([x["false_cull_ratio"] for x in per_view_stats], dtype=np.float64)
    false_keep_ratio_list = np.array([x["false_keep_ratio"] for x in per_view_stats], dtype=np.float64)
    mismatch_ratio_list = np.array([x["mismatch_ratio"] for x in per_view_stats], dtype=np.float64)

    summary = {
        "scene_name": os.path.basename(model_path),
        "model_path": model_path,
        "source_path": source_path,
        "total_gaussians": total_gaussians,
        "num_video_views": num_video_views,
        "false_cull": {
            "mean_count": float(np.mean(false_cull_list)),
            "std_count": float(np.std(false_cull_list)),
            "min_count": int(np.min(false_cull_list)),
            "max_count": int(np.max(false_cull_list)),
            "mean_ratio": float(np.mean(false_cull_ratio_list)),
            "std_ratio": float(np.std(false_cull_ratio_list)),
        },
        "false_keep": {
            "mean_count": float(np.mean(false_keep_list)),
            "std_count": float(np.std(false_keep_list)),
            "min_count": int(np.min(false_keep_list)),
            "max_count": int(np.max(false_keep_list)),
            "mean_ratio": float(np.mean(false_keep_ratio_list)),
            "std_ratio": float(np.std(false_keep_ratio_list)),
        },
        "mismatch_total": {
            "mean_count": float(np.mean(mismatch_list)),
            "std_count": float(np.std(mismatch_list)),
            "min_count": int(np.min(mismatch_list)),
            "max_count": int(np.max(mismatch_list)),
            "mean_ratio": float(np.mean(mismatch_ratio_list)),
            "std_ratio": float(np.std(mismatch_ratio_list)),
        },
        "per_view_stats": per_view_stats,
    }

    return summary


def save_scene_results(summary: dict, output_dir: str):
    """保存单场景json结果。"""
    scene_name = summary["scene_name"]
    scene_output_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    json_path = os.path.join(scene_output_dir, "frustum_culling_error_stats.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"场景结果已保存: {json_path}")


def generate_dataset_summary(all_summaries, dataset_name: str, output_dir: str):
    """
    生成数据集级统计与可视化：
    - 横轴：scene
    - 纵轴：错误被cull / 错误没被cull 的平均数量与平均占比
    """
    dataset_output_dir = os.path.join(output_dir, "aggregated_analysis")
    os.makedirs(dataset_output_dir, exist_ok=True)

    scene_names = [s["scene_name"] for s in all_summaries]
    false_cull_mean_counts = [s["false_cull"]["mean_count"] for s in all_summaries]
    false_keep_mean_counts = [s["false_keep"]["mean_count"] for s in all_summaries]
    false_cull_mean_ratios = [s["false_cull"]["mean_ratio"] * 100.0 for s in all_summaries]
    false_keep_mean_ratios = [s["false_keep"]["mean_ratio"] * 100.0 for s in all_summaries]

    # 1) 可视化：两个子图，分别展示数量和占比
    x = np.arange(len(scene_names))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(scene_names) * 1.2), 10), sharex=True)
    fig.suptitle(f"Frustum Culling Error Analysis - Dataset: {dataset_name}", fontsize=16, fontweight="bold")

    # 数量
    axes[0].bar(x - width / 2, false_cull_mean_counts, width, label="错误被cull (false_cull)", color="#e74c3c", alpha=0.85)
    axes[0].bar(x + width / 2, false_keep_mean_counts, width, label="错误没被cull (false_keep)", color="#3498db", alpha=0.85)
    axes[0].set_ylabel("平均错误高斯球数量 / view")
    axes[0].set_title("按场景：错误culling数量")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="upper right")

    # 占比
    axes[1].bar(x - width / 2, false_cull_mean_ratios, width, label="错误被cull占比", color="#e67e22", alpha=0.85)
    axes[1].bar(x + width / 2, false_keep_mean_ratios, width, label="错误没被cull占比", color="#2ecc71", alpha=0.85)
    axes[1].set_ylabel("平均占比 (%) / view")
    axes[1].set_title("按场景：错误culling占比")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(loc="upper right")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scene_names, rotation=45, ha="right")
    axes[1].set_xlabel("Scene")

    plt.tight_layout()
    plot_path = os.path.join(
        dataset_output_dir,
        f"{dataset_name.replace('/', '_')}_frustum_culling_error_aggregated.png",
    )
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"数据集可视化已保存: {plot_path}")

    # 2) 输出CSV，便于后续二次分析
    csv_path = os.path.join(
        dataset_output_dir,
        f"{dataset_name.replace('/', '_')}_frustum_culling_error_aggregated.csv",
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scene",
                "total_gaussians",
                "num_video_views",
                "false_cull_mean_count",
                "false_keep_mean_count",
                "mismatch_mean_count",
                "false_cull_mean_ratio",
                "false_keep_mean_ratio",
                "mismatch_mean_ratio",
            ]
        )
        for s in all_summaries:
            writer.writerow(
                [
                    s["scene_name"],
                    s["total_gaussians"],
                    s["num_video_views"],
                    s["false_cull"]["mean_count"],
                    s["false_keep"]["mean_count"],
                    s["mismatch_total"]["mean_count"],
                    s["false_cull"]["mean_ratio"],
                    s["false_keep"]["mean_ratio"],
                    s["mismatch_total"]["mean_ratio"],
                ]
            )
    print(f"数据集汇总CSV已保存: {csv_path}")

    # 3) 汇总JSON
    dataset_stats = {
        "dataset": dataset_name,
        "num_scenes": len(all_summaries),
        "scenes": {},
        "dataset_level": {
            "false_cull_mean_count": float(np.mean(false_cull_mean_counts)),
            "false_keep_mean_count": float(np.mean(false_keep_mean_counts)),
            "false_cull_mean_ratio": float(np.mean([x / 100.0 for x in false_cull_mean_ratios])),
            "false_keep_mean_ratio": float(np.mean([x / 100.0 for x in false_keep_mean_ratios])),
        },
        "artifacts": {
            "plot": plot_path,
            "csv": csv_path,
        },
    }
    for s in all_summaries:
        dataset_stats["scenes"][s["scene_name"]] = {
            "total_gaussians": s["total_gaussians"],
            "num_video_views": s["num_video_views"],
            "false_cull": s["false_cull"],
            "false_keep": s["false_keep"],
            "mismatch_total": s["mismatch_total"],
        }

    json_path = os.path.join(
        dataset_output_dir,
        f"{dataset_name.replace('/', '_')}_frustum_culling_error_aggregated.json",
    )
    with open(json_path, "w") as f:
        json.dump(dataset_stats, f, indent=2)
    print(f"数据集汇总JSON已保存: {json_path}")


def aggregate_dataset_results(base_dir: str, dataset_name: str, iteration: int, output_dir: str):
    """遍历并分析数据集内所有场景。"""
    dataset_dir = os.path.join(base_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        print(f"错误: 数据集目录不存在: {dataset_dir}")
        return

    scene_names = []
    for item in sorted(os.listdir(dataset_dir)):
        scene_path = os.path.join(dataset_dir, item)
        if os.path.isdir(scene_path) and os.path.exists(os.path.join(scene_path, "point_cloud")):
            scene_names.append(item)

    print(f"\n找到 {len(scene_names)} 个场景: {scene_names}")

    all_summaries = []
    for scene_name in scene_names:
        model_path = os.path.join(dataset_dir, scene_name)
        source_path = _resolve_source_path(model_path, dataset_name, scene_name)
        if source_path is None:
            print(f"警告: 无法确定 {scene_name} 的source_path，跳过")
            continue

        try:
            summary = analyze_scene_frustum_culling_error(model_path, source_path, iteration)
            save_scene_results(summary, output_dir)
            all_summaries.append(summary)
        except Exception as exc:
            print(f"错误: 分析场景 {scene_name} 失败: {exc}")
            import traceback

            traceback.print_exc()

    if not all_summaries:
        print("没有成功分析任何场景。")
        return

    generate_dataset_summary(all_summaries, dataset_name, output_dir)


if __name__ == "__main__":
    parser = ArgumentParser(description="Frustum Culling Error Analysis Script")
    parser.add_argument("--dataset", type=str, default="", help="数据集名称（分析整个数据集）")
    parser.add_argument("--base_dir", type=str, default="output", help="模型输出根目录")
    parser.add_argument("--iteration", type=int, default=-1, help="加载迭代次数（-1表示最新）")
    parser.add_argument("--output_dir", type=str, default="output", help="分析结果输出根目录")
    args = parser.parse_args()

    safe_state(False)

    if not args.dataset:
        print("错误: 当前脚本用于数据集级分析，请提供 --dataset")
        parser.print_help()
        sys.exit(1)

    output_dir = os.path.join(args.output_dir, args.dataset, "frustum_culling_error_analysis")
    os.makedirs(output_dir, exist_ok=True)
    aggregate_dataset_results(args.base_dir, args.dataset, args.iteration, output_dir)
