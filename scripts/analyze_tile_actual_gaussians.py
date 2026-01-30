#!/usr/bin/env python3
"""
分析4DGS中各个场景test split view中各tile实际处理的高斯球数量统计（考虑early stop）

用法:
    # 分析单个场景
    python scripts/analyze_tile_actual_gaussians.py --model_path output/dynerf/bouncingballs --source_path data/dynerf/bouncingballs --iteration 30000
    
    # 分析整个数据集的所有场景
    python scripts/analyze_tile_actual_gaussians.py --dataset dynerf --base_dir output --iteration 30000
"""

import os
import sys

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
import torch
from tqdm import tqdm
import math

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from utils.general_utils import safe_state

# Tile大小（通常Gaussian Splatting使用16x16的tile）
TILE_SIZE = 16


def compute_tile_actual_gaussians(n_contrib, image_width, image_height, tile_size=TILE_SIZE):
    """
    计算每个tile实际处理的高斯球数量（考虑early stop）
    
    Args:
        n_contrib: 每个像素实际处理的高斯球数量 (H, W)
        image_width: 图像宽度
        image_height: 图像高度
        tile_size: tile大小（默认16x16）
    
    Returns:
        tile_stats: dict，包含统计信息
    """
    # 转换为numpy数组
    if isinstance(n_contrib, torch.Tensor):
        n_contrib_np = n_contrib.cpu().numpy()
    else:
        n_contrib_np = np.array(n_contrib)
    
    # 计算tile网格大小
    num_tiles_x = math.ceil(image_width / tile_size)
    num_tiles_y = math.ceil(image_height / tile_size)
    
    # 计算每个tile的平均处理高斯球数量
    tile_avg_gaussians = []
    tile_max_gaussians = []
    tile_min_gaussians = []
    
    for tile_y in range(num_tiles_y):
        for tile_x in range(num_tiles_x):
            # 计算tile的像素范围
            pix_min_x = tile_x * tile_size
            pix_max_x = min((tile_x + 1) * tile_size, image_width)
            pix_min_y = tile_y * tile_size
            pix_max_y = min((tile_y + 1) * tile_size, image_height)
            
            # 提取该tile的n_contrib值
            tile_n_contrib = n_contrib_np[pix_min_y:pix_max_y, pix_min_x:pix_max_x]
            
            # 只考虑有效像素（n_contrib > 0）
            valid_mask = tile_n_contrib > 0
            if valid_mask.sum() > 0:
                valid_n_contrib = tile_n_contrib[valid_mask]
                tile_avg_gaussians.append(float(np.mean(valid_n_contrib)))
                tile_max_gaussians.append(int(np.max(valid_n_contrib)))
                tile_min_gaussians.append(int(np.min(valid_n_contrib)))
            else:
                # 如果tile内没有有效像素，跳过
                pass
    
    tile_stats = {
        "num_tiles": len(tile_avg_gaussians),
        "total_tiles": num_tiles_x * num_tiles_y,
        "tile_grid": (num_tiles_x, num_tiles_y),
        "image_size": (image_width, image_height),
        "tile_avg_gaussians": tile_avg_gaussians,
        "tile_max_gaussians": tile_max_gaussians,
        "tile_min_gaussians": tile_min_gaussians
    }
    
    if len(tile_avg_gaussians) > 0:
        tile_stats["max_gaussians_per_tile"] = int(np.max(tile_max_gaussians))
        tile_stats["min_gaussians_per_tile"] = int(np.min(tile_min_gaussians))
        tile_stats["mean_gaussians_per_tile"] = float(np.mean(tile_avg_gaussians))
        tile_stats["median_gaussians_per_tile"] = float(np.median(tile_avg_gaussians))
        tile_stats["std_gaussians_per_tile"] = float(np.std(tile_avg_gaussians))
    else:
        tile_stats["max_gaussians_per_tile"] = 0
        tile_stats["min_gaussians_per_tile"] = 0
        tile_stats["mean_gaussians_per_tile"] = 0.0
        tile_stats["median_gaussians_per_tile"] = 0.0
        tile_stats["std_gaussians_per_tile"] = 0.0
    
    return tile_stats


def analyze_scene_tile_actual_gaussians(model_path, source_path, iteration, dataset_type=None, frame_idx=None):
    """
    分析单个场景的tile实际处理高斯球统计
    
    Args:
        model_path: 模型路径
        source_path: 数据源路径
        iteration: 迭代次数
        dataset_type: 数据集类型（可选）
        frame_idx: 要分析的帧索引（如果为None，分析所有test view）
    
    Returns:
        dict: 包含统计信息的字典
    """
    print(f"\n分析场景: {model_path}")
    
    # 创建参数解析器并设置参数
    parser = ArgumentParser()
    model_params_class = ModelParams(parser, sentinel=True)
    pipeline_params_class = PipelineParams(parser)
    hyperparam_class = ModelHiddenParams(parser)
    
    from argparse import Namespace
    
    # 默认参数
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
    
    # 尝试从保存的配置文件中加载参数
    cfg_path = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r') as cfg_file:
                cfgfile_string = cfg_file.read()
            args_cfgfile = eval(cfgfile_string)
            for key, value in vars(args_cfgfile).items():
                if key not in ['model_path', 'source_path']:
                    setattr(args, key, value)
        except Exception as e:
            print(f"警告: 无法加载保存的配置，使用默认值: {e}")
    
    # 提取参数组
    model_params = model_params_class.extract(args)
    pipeline_params = pipeline_params_class.extract(args)
    hyperparam = hyperparam_class.extract(args)
    
    # 加载场景
    gaussians = GaussianModel(model_params.sh_degree, hyperparam)
    scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)
    
    # 获取测试相机
    test_cameras = scene.getVideoCameras()
    num_test_views = len(test_cameras)
    
    print(f"测试视图数量: {num_test_views}")
    print(f"总高斯球数量: {gaussians.get_xyz.shape[0]}")
    
    # 设置背景色
    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 统计信息
    total_gaussians = gaussians.get_xyz.shape[0]
    view_stats = []
    
    # 确定要分析的视图
    if frame_idx is not None:
        views_to_analyze = [test_cameras[frame_idx]] if frame_idx < len(test_cameras) else []
        view_indices = [frame_idx]
    else:
        views_to_analyze = test_cameras
        view_indices = list(range(len(test_cameras)))
    
    # 对每个test camera view进行分析
    for idx, view in zip(view_indices, tqdm(views_to_analyze, desc="分析tile实际处理高斯球统计")):
        # 使用stage="coarse"跳过deformation，直接使用canonical高斯球
        rendering_results = render(
            view, 
            gaussians, 
            pipeline_params, 
            background, 
            stage="coarse",  # 跳过deformation
            cam_type=scene.dataset_type
        )
        
        # 获取n_contrib（每个像素实际处理的高斯球数量）
        if "n_contrib" not in rendering_results:
            print(f"警告: 视图 {idx} 没有n_contrib信息，可能需要重新编译CUDA扩展")
            continue
        
        n_contrib = rendering_results["n_contrib"]
        
        # 计算每个tile的实际处理高斯球数量
        tile_stats = compute_tile_actual_gaussians(
            n_contrib,
            int(view.image_width),
            int(view.image_height),
            tile_size=TILE_SIZE
        )
        
        view_stats.append({
            "view_idx": idx,
            "image_width": int(view.image_width),
            "image_height": int(view.image_height),
            "total_gaussians": total_gaussians,
            "tile_stats": tile_stats
        })
    
    if len(view_stats) == 0:
        print("错误: 没有成功分析任何视图")
        return None
    
    # 计算汇总统计
    all_max = [s["tile_stats"]["max_gaussians_per_tile"] for s in view_stats]
    all_min = [s["tile_stats"]["min_gaussians_per_tile"] for s in view_stats]
    all_mean = [s["tile_stats"]["mean_gaussians_per_tile"] for s in view_stats]
    all_median = [s["tile_stats"]["median_gaussians_per_tile"] for s in view_stats]
    
    summary = {
        "scene_name": os.path.basename(model_path),
        "total_gaussians": total_gaussians,
        "num_test_views": num_test_views,
        "num_views_analyzed": len(view_stats),
        "tile_size": TILE_SIZE,
        "per_view_stats": view_stats,
        "aggregated_stats": {
            "max_gaussians_per_tile": {
                "max": int(np.max(all_max)) if len(all_max) > 0 else 0,
                "min": int(np.min(all_max)) if len(all_max) > 0 else 0,
                "mean": float(np.mean(all_max)) if len(all_max) > 0 else 0.0,
                "median": float(np.median(all_max)) if len(all_max) > 0 else 0.0
            },
            "min_gaussians_per_tile": {
                "max": int(np.max(all_min)) if len(all_min) > 0 else 0,
                "min": int(np.min(all_min)) if len(all_min) > 0 else 0,
                "mean": float(np.mean(all_min)) if len(all_min) > 0 else 0.0,
                "median": float(np.median(all_min)) if len(all_min) > 0 else 0.0
            },
            "mean_gaussians_per_tile": {
                "max": float(np.max(all_mean)) if len(all_mean) > 0 else 0.0,
                "min": float(np.min(all_mean)) if len(all_mean) > 0 else 0.0,
                "mean": float(np.mean(all_mean)) if len(all_mean) > 0 else 0.0,
                "median": float(np.median(all_mean)) if len(all_mean) > 0 else 0.0
            },
            "median_gaussians_per_tile": {
                "max": float(np.max(all_median)) if len(all_median) > 0 else 0.0,
                "min": float(np.min(all_median)) if len(all_median) > 0 else 0.0,
                "mean": float(np.mean(all_median)) if len(all_median) > 0 else 0.0,
                "median": float(np.median(all_median)) if len(all_median) > 0 else 0.0
            }
        }
    }
    
    return summary


def save_scene_results(summary, output_dir):
    """保存单个场景的结果"""
    scene_name = summary["scene_name"]
    scene_output_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    
    # 保存JSON结果
    json_path = os.path.join(scene_output_dir, "tile_actual_gaussians_stats.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"结果已保存到: {json_path}")
    return json_path


def plot_scene_results(summary, output_dir):
    """为单个场景生成图表"""
    scene_name = summary["scene_name"]
    scene_output_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    
    per_view_stats = summary["per_view_stats"]
    view_indices = [s["view_idx"] for s in per_view_stats]
    
    # 提取统计数据
    max_per_view = [s["tile_stats"]["max_gaussians_per_tile"] for s in per_view_stats]
    min_per_view = [s["tile_stats"]["min_gaussians_per_tile"] for s in per_view_stats]
    mean_per_view = [s["tile_stats"]["mean_gaussians_per_tile"] for s in per_view_stats]
    median_per_view = [s["tile_stats"]["median_gaussians_per_tile"] for s in per_view_stats]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Tile Actual Gaussians Analysis (with Early Stop): {scene_name}', fontsize=16, fontweight='bold')
    
    # 1. 每个view的最大高斯球数量
    axes[0, 0].plot(view_indices, max_per_view, 'r-o', linewidth=1.5, markersize=4, alpha=0.7)
    axes[0, 0].axhline(y=summary["aggregated_stats"]["max_gaussians_per_tile"]["mean"], 
                       color='r', linestyle='--', 
                       label=f'Mean: {summary["aggregated_stats"]["max_gaussians_per_tile"]["mean"]:.0f}')
    axes[0, 0].set_xlabel('View Index')
    axes[0, 0].set_ylabel('Max Gaussians per Tile')
    axes[0, 0].set_title('Max Gaussians per Tile per View (Actual)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 每个view的平均高斯球数量
    axes[0, 1].plot(view_indices, mean_per_view, 'g-o', linewidth=1.5, markersize=4, alpha=0.7)
    axes[0, 1].axhline(y=summary["aggregated_stats"]["mean_gaussians_per_tile"]["mean"], 
                       color='g', linestyle='--',
                       label=f'Mean: {summary["aggregated_stats"]["mean_gaussians_per_tile"]["mean"]:.2f}')
    axes[0, 1].set_xlabel('View Index')
    axes[0, 1].set_ylabel('Mean Gaussians per Tile')
    axes[0, 1].set_title('Mean Gaussians per Tile per View (Actual)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 每个view的最小高斯球数量
    axes[1, 0].plot(view_indices, min_per_view, 'b-o', linewidth=1.5, markersize=4, alpha=0.7)
    axes[1, 0].axhline(y=summary["aggregated_stats"]["min_gaussians_per_tile"]["mean"], 
                       color='b', linestyle='--',
                       label=f'Mean: {summary["aggregated_stats"]["min_gaussians_per_tile"]["mean"]:.0f}')
    axes[1, 0].set_xlabel('View Index')
    axes[1, 0].set_ylabel('Min Gaussians per Tile')
    axes[1, 0].set_title('Min Gaussians per Tile per View (Actual)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 统计摘要
    axes[1, 1].axis('off')
    stats_text = f"""
统计摘要 (考虑Early Stop):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总高斯球数量: {summary["total_gaussians"]:,}
测试视图数量: {summary["num_test_views"]}
分析的视图数量: {summary["num_views_analyzed"]}
Tile大小: {summary["tile_size"]}x{summary["tile_size"]}

每个Tile的最大高斯球数量 (实际):
  最大值: {summary["aggregated_stats"]["max_gaussians_per_tile"]["max"]:,}
  最小值: {summary["aggregated_stats"]["max_gaussians_per_tile"]["min"]:,}
  平均值: {summary["aggregated_stats"]["max_gaussians_per_tile"]["mean"]:.0f}
  中位数: {summary["aggregated_stats"]["max_gaussians_per_tile"]["median"]:.0f}

每个Tile的平均高斯球数量 (实际):
  最大值: {summary["aggregated_stats"]["mean_gaussians_per_tile"]["max"]:.2f}
  最小值: {summary["aggregated_stats"]["mean_gaussians_per_tile"]["min"]:.2f}
  平均值: {summary["aggregated_stats"]["mean_gaussians_per_tile"]["mean"]:.2f}
  中位数: {summary["aggregated_stats"]["mean_gaussians_per_tile"]["median"]:.2f}

每个Tile的最小高斯球数量 (实际):
  最大值: {summary["aggregated_stats"]["min_gaussians_per_tile"]["max"]:,}
  最小值: {summary["aggregated_stats"]["min_gaussians_per_tile"]["min"]:,}
  平均值: {summary["aggregated_stats"]["min_gaussians_per_tile"]["mean"]:.0f}
  中位数: {summary["aggregated_stats"]["min_gaussians_per_tile"]["median"]:.0f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plot_path = os.path.join(scene_output_dir, "tile_actual_gaussians_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存到: {plot_path}")


def aggregate_dataset_results(base_dir, dataset_name, iteration, output_dir, frame_idx=None):
    """
    汇总整个数据集的所有场景结果
    """
    dataset_dir = os.path.join(base_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        print(f"错误: 数据集目录不存在: {dataset_dir}")
        return
    
    # 查找所有场景
    scenes = []
    for item in os.listdir(dataset_dir):
        scene_path = os.path.join(dataset_dir, item)
        if os.path.isdir(scene_path):
            point_cloud_dir = os.path.join(scene_path, "point_cloud")
            if os.path.exists(point_cloud_dir):
                scenes.append(item)
    
    print(f"\n找到 {len(scenes)} 个场景: {scenes}")
    
    # 分析每个场景
    all_summaries = []
    for scene in scenes:
        model_path = os.path.join(dataset_dir, scene)
        source_path = None
        cfg_path = os.path.join(model_path, "cfg_args")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r') as f:
                    cfg_content = f.read()
                    if 'source_path' in cfg_content:
                        import re
                        match = re.search(r"source_path['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", cfg_content)
                        if match:
                            source_path = match.group(1)
            except:
                pass
        
        if source_path is None:
            data_dir = os.path.join("data", dataset_name, scene)
            if os.path.exists(data_dir):
                source_path = data_dir
            else:
                print(f"警告: 无法找到场景 {scene} 的source_path，跳过")
                continue
        
        try:
            summary = analyze_scene_tile_actual_gaussians(model_path, source_path, iteration, frame_idx=frame_idx)
            if summary is not None:
                save_scene_results(summary, output_dir)
                plot_scene_results(summary, output_dir)
                all_summaries.append(summary)
        except Exception as e:
            print(f"错误: 分析场景 {scene} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    if len(all_summaries) == 0:
        print("没有成功分析任何场景")
        return
    
    # 生成数据集级别的汇总
    generate_dataset_summary(all_summaries, dataset_name, output_dir)


def generate_dataset_summary(all_summaries, dataset_name, output_dir):
    """生成数据集级别的汇总图表"""
    print(f"\n生成数据集级别的汇总...")
    
    dataset_output_dir = os.path.join(output_dir, "aggregated_analysis")
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # 准备数据
    scene_names = [s["scene_name"] for s in all_summaries]
    
    # 提取每个场景的统计信息
    max_values = [s["aggregated_stats"]["max_gaussians_per_tile"]["max"] for s in all_summaries]
    mean_values = [s["aggregated_stats"]["mean_gaussians_per_tile"]["mean"] for s in all_summaries]
    min_values = [s["aggregated_stats"]["min_gaussians_per_tile"]["mean"] for s in all_summaries]
    
    # 创建簇状柱状图
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle(f'Tile Actual Gaussians Analysis (with Early Stop) - Dataset: {dataset_name}', fontsize=16, fontweight='bold')
    
    # 设置x轴位置
    x = np.arange(len(scene_names))
    width = 0.25  # 每个柱子的宽度
    
    # 创建三个柱子组（max, mean, min）
    bars1 = ax.bar(x - width, max_values, width, label='Max', alpha=0.8, color='#e74c3c')
    bars2 = ax.bar(x, mean_values, width, label='Mean', alpha=0.8, color='#3498db')
    bars3 = ax.bar(x + width, min_values, width, label='Min', alpha=0.8, color='#2ecc71')
    
    # 设置对数坐标
    ax.set_yscale('log')
    
    # 设置标签和标题
    ax.set_xlabel('Scene', fontsize=12)
    ax.set_ylabel('Gaussians per Tile (Log Scale)', fontsize=12)
    ax.set_title('Actual Gaussians per Tile Statistics (Max, Mean, Min) by Scene', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scene_names, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', which='both')
    
    # 在柱子上添加数值标签（可选，如果数值不是太大）
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height) if height >= 1 else f"{height:.2f}"}',
                       ha='center', va='bottom', fontsize=8, rotation=0)
    
    # 根据数值范围决定是否添加标签
    max_val = max(max(max_values), max(mean_values), max(min_values))
    if max_val < 10000:  # 如果数值不太大，添加标签
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
    
    plt.tight_layout()
    plot_path = os.path.join(dataset_output_dir, f"{dataset_name.replace('/', '_')}_tile_actual_gaussians_aggregated.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"数据集汇总图表已保存到: {plot_path}")
    
    # 保存汇总统计
    aggregated_stats = {
        "dataset": dataset_name,
        "num_scenes": len(all_summaries),
        "tile_size": TILE_SIZE,
        "scenes": {}
    }
    
    for summary in all_summaries:
        scene_name = summary["scene_name"]
        aggregated_stats["scenes"][scene_name] = {
            "total_gaussians": summary["total_gaussians"],
            "num_test_views": summary["num_test_views"],
            "max_gaussians_per_tile": {
                "max": summary["aggregated_stats"]["max_gaussians_per_tile"]["max"],
                "mean": summary["aggregated_stats"]["max_gaussians_per_tile"]["mean"]
            },
            "mean_gaussians_per_tile": {
                "mean": summary["aggregated_stats"]["mean_gaussians_per_tile"]["mean"]
            },
            "min_gaussians_per_tile": {
                "mean": summary["aggregated_stats"]["min_gaussians_per_tile"]["mean"]
            }
        }
    
    # 计算数据集级别的总体统计
    aggregated_stats["dataset_level"] = {
        "max_gaussians_per_tile": {
            "max": float(np.max(max_values)),
            "min": float(np.min(max_values)),
            "mean": float(np.mean(max_values)),
            "std": float(np.std(max_values))
        },
        "mean_gaussians_per_tile": {
            "max": float(np.max(mean_values)),
            "min": float(np.min(mean_values)),
            "mean": float(np.mean(mean_values)),
            "std": float(np.std(mean_values))
        },
        "min_gaussians_per_tile": {
            "max": float(np.max(min_values)),
            "min": float(np.min(min_values)),
            "mean": float(np.mean(min_values)),
            "std": float(np.std(min_values))
        }
    }
    
    json_path = os.path.join(dataset_output_dir, f"{dataset_name.replace('/', '_')}_tile_actual_gaussians_aggregated.json")
    with open(json_path, 'w') as f:
        json.dump(aggregated_stats, f, indent=2)
    
    print(f"数据集汇总统计已保存到: {json_path}")
    
    # 打印汇总信息
    print("\n" + "="*60)
    print(f"数据集汇总统计: {dataset_name}")
    print("="*60)
    print(f"场景数量: {len(all_summaries)}")
    print(f"\n每个Tile的最大高斯球数量 (实际):")
    print(f"  最大值: {aggregated_stats['dataset_level']['max_gaussians_per_tile']['max']:.0f}")
    print(f"  最小值: {aggregated_stats['dataset_level']['max_gaussians_per_tile']['min']:.0f}")
    print(f"  平均值: {aggregated_stats['dataset_level']['max_gaussians_per_tile']['mean']:.0f} ± {aggregated_stats['dataset_level']['max_gaussians_per_tile']['std']:.0f}")
    print(f"\n每个Tile的平均高斯球数量 (实际):")
    print(f"  最大值: {aggregated_stats['dataset_level']['mean_gaussians_per_tile']['max']:.2f}")
    print(f"  最小值: {aggregated_stats['dataset_level']['mean_gaussians_per_tile']['min']:.2f}")
    print(f"  平均值: {aggregated_stats['dataset_level']['mean_gaussians_per_tile']['mean']:.2f} ± {aggregated_stats['dataset_level']['mean_gaussians_per_tile']['std']:.2f}")
    print(f"\n每个Tile的最小高斯球数量 (实际):")
    print(f"  最大值: {aggregated_stats['dataset_level']['min_gaussians_per_tile']['max']:.0f}")
    print(f"  最小值: {aggregated_stats['dataset_level']['min_gaussians_per_tile']['min']:.0f}")
    print(f"  平均值: {aggregated_stats['dataset_level']['min_gaussians_per_tile']['mean']:.2f} ± {aggregated_stats['dataset_level']['min_gaussians_per_tile']['std']:.2f}")
    print("="*60)


if __name__ == "__main__":
    parser = ArgumentParser(description="Tile Actual Gaussians Analysis Script (with Early Stop)")
    parser.add_argument("--model_path", type=str, default="", help="单个场景的模型路径")
    parser.add_argument("--source_path", type=str, default="", help="单个场景的数据源路径")
    parser.add_argument("--dataset", type=str, default="", help="数据集名称（用于分析整个数据集）")
    parser.add_argument("--base_dir", type=str, default="output", help="基础输出目录")
    parser.add_argument("--iteration", type=int, default=-1, help="迭代次数")
    parser.add_argument("--output_dir", type=str, default="output", help="结果输出目录")
    parser.add_argument("--frame_idx", type=int, default=None, help="要分析的帧索引（如果为None，分析所有test view）")
    
    args = parser.parse_args()
    
    # 初始化系统状态
    safe_state(False)
    
    if args.dataset:
        # 分析整个数据集
        output_dir = os.path.join(args.output_dir, args.dataset, "tile_actual_gaussians_analysis")
        os.makedirs(output_dir, exist_ok=True)
        aggregate_dataset_results(args.base_dir, args.dataset, args.iteration, output_dir, frame_idx=args.frame_idx)
    elif args.model_path and args.source_path:
        # 分析单个场景
        output_dir = os.path.join(args.output_dir, "tile_actual_gaussians_analysis")
        os.makedirs(output_dir, exist_ok=True)
        summary = analyze_scene_tile_actual_gaussians(args.model_path, args.source_path, args.iteration, frame_idx=args.frame_idx)
        if summary is not None:
            save_scene_results(summary, output_dir)
            plot_scene_results(summary, output_dir)
    else:
        print("错误: 请提供 --model_path 和 --source_path（单个场景）或 --dataset（整个数据集）")
        parser.print_help()
