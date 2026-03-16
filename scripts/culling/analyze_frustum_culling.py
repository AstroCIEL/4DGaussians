"""
分析Frustum Culling统计脚本
分析某数据集某场景中，所有test camera view的frustum culling环节有多少高斯球被剔除
（使用canonical高斯球，不考虑deformation）

用法:
    # 分析单个场景
    python scripts/analyze_frustum_culling.py --model_path output/dynerf/bouncingballs --source_path data/dynerf/bouncingballs --iteration 30000
    
    # 分析整个数据集的所有场景
    python scripts/analyze_frustum_culling.py --dataset dynerf --base_dir output --iteration 30000
"""

import os
import sys

# 添加项目根目录到Python路径，确保可以导入模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
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

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from utils.general_utils import safe_state


def analyze_scene_frustum_culling(model_path, source_path, iteration, dataset_type=None):
    """
    分析单个场景的frustum culling统计
    
    Args:
        model_path: 模型路径
        source_path: 数据源路径
        iteration: 迭代次数
        dataset_type: 数据集类型（可选，会自动检测）
    
    Returns:
        dict: 包含统计信息的字典
    """
    print(f"\n分析场景: {model_path}")
    
    # 创建参数解析器并设置参数
    parser = ArgumentParser()
    model_params_class = ModelParams(parser, sentinel=True)
    pipeline_params_class = PipelineParams(parser)
    hyperparam_class = ModelHiddenParams(parser)
    
    # 创建命名空间对象来存储参数
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
            # 合并配置文件的参数，但保留我们提供的 model_path 和 source_path
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
    culling_stats = []
    
    # 对每个test camera view进行分析
    for idx, view in enumerate(tqdm(test_cameras, desc="分析frustum culling")):
        # 使用stage="coarse"跳过deformation，直接使用canonical高斯球
        rendering_results = render(
            view, 
            gaussians, 
            pipeline_params, 
            background, 
            stage="coarse",  # 跳过deformation
            cam_type=scene.dataset_type
        )
        
        radii = rendering_results["radii"]
        visibility_filter = rendering_results["visibility_filter"]
        
        # 统计被剔除的高斯球
        num_culled = (radii == 0).sum().item()
        num_visible = visibility_filter.sum().item()
        
        culling_stats.append({
            "view_idx": idx,
            "total_gaussians": total_gaussians,
            "culled_gaussians": num_culled,
            "visible_gaussians": num_visible,
            "culling_ratio": num_culled / total_gaussians if total_gaussians > 0 else 0.0,
            "visibility_ratio": num_visible / total_gaussians if total_gaussians > 0 else 0.0
        })
    
    # 计算汇总统计
    culled_list = [s["culled_gaussians"] for s in culling_stats]
    visible_list = [s["visible_gaussians"] for s in culling_stats]
    culling_ratio_list = [s["culling_ratio"] for s in culling_stats]
    
    summary = {
        "scene_name": os.path.basename(model_path),
        "total_gaussians": total_gaussians,
        "num_test_views": num_test_views,
        "culled_gaussians": {
            "mean": float(np.mean(culled_list)),
            "std": float(np.std(culled_list)),
            "min": int(np.min(culled_list)),
            "max": int(np.max(culled_list)),
            "median": float(np.median(culled_list))
        },
        "visible_gaussians": {
            "mean": float(np.mean(visible_list)),
            "std": float(np.std(visible_list)),
            "min": int(np.min(visible_list)),
            "max": int(np.max(visible_list)),
            "median": float(np.median(visible_list))
        },
        "culling_ratio": {
            "mean": float(np.mean(culling_ratio_list)),
            "std": float(np.std(culling_ratio_list)),
            "min": float(np.min(culling_ratio_list)),
            "max": float(np.max(culling_ratio_list)),
            "median": float(np.median(culling_ratio_list))
        },
        "per_view_stats": culling_stats
    }
    
    return summary


def save_scene_results(summary, output_dir):
    """保存单个场景的结果"""
    scene_name = summary["scene_name"]
    scene_output_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    
    # 保存JSON结果
    json_path = os.path.join(scene_output_dir, "frustum_culling_stats.json")
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
    culled = [s["culled_gaussians"] for s in per_view_stats]
    visible = [s["visible_gaussians"] for s in per_view_stats]
    culling_ratio = [s["culling_ratio"] * 100 for s in per_view_stats]  # 转换为百分比
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Frustum Culling Analysis: {scene_name}', fontsize=16)
    
    # 1. 被剔除的高斯球数量
    axes[0, 0].plot(view_indices, culled, 'r-', linewidth=1.5, alpha=0.7)
    axes[0, 0].axhline(y=summary["culled_gaussians"]["mean"], color='r', linestyle='--', 
                       label=f'Mean: {summary["culled_gaussians"]["mean"]:.0f}')
    axes[0, 0].set_xlabel('View Index')
    axes[0, 0].set_ylabel('Culled Gaussians')
    axes[0, 0].set_title('Culled Gaussians per View')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 可见的高斯球数量
    axes[0, 1].plot(view_indices, visible, 'g-', linewidth=1.5, alpha=0.7)
    axes[0, 1].axhline(y=summary["visible_gaussians"]["mean"], color='g', linestyle='--',
                       label=f'Mean: {summary["visible_gaussians"]["mean"]:.0f}')
    axes[0, 1].set_xlabel('View Index')
    axes[0, 1].set_ylabel('Visible Gaussians')
    axes[0, 1].set_title('Visible Gaussians per View')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 剔除比例
    axes[1, 0].plot(view_indices, culling_ratio, 'b-', linewidth=1.5, alpha=0.7)
    axes[1, 0].axhline(y=summary["culling_ratio"]["mean"] * 100, color='b', linestyle='--',
                       label=f'Mean: {summary["culling_ratio"]["mean"]*100:.2f}%')
    axes[1, 0].set_xlabel('View Index')
    axes[1, 0].set_ylabel('Culling Ratio (%)')
    axes[1, 0].set_title('Culling Ratio per View')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 统计摘要
    axes[1, 1].axis('off')
    stats_text = f"""
统计摘要:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总高斯球数量: {summary["total_gaussians"]:,}
测试视图数量: {summary["num_test_views"]}

被剔除的高斯球:
  平均值: {summary["culled_gaussians"]["mean"]:.0f}
  标准差: {summary["culled_gaussians"]["std"]:.0f}
  最小值: {summary["culled_gaussians"]["min"]:,}
  最大值: {summary["culled_gaussians"]["max"]:,}
  中位数: {summary["culled_gaussians"]["median"]:.0f}

可见的高斯球:
  平均值: {summary["visible_gaussians"]["mean"]:.0f}
  标准差: {summary["visible_gaussians"]["std"]:.0f}
  最小值: {summary["visible_gaussians"]["min"]:,}
  最大值: {summary["visible_gaussians"]["max"]:,}
  中位数: {summary["visible_gaussians"]["median"]:.0f}

剔除比例:
  平均值: {summary["culling_ratio"]["mean"]*100:.2f}%
  标准差: {summary["culling_ratio"]["std"]*100:.2f}%
  最小值: {summary["culling_ratio"]["min"]*100:.2f}%
  最大值: {summary["culling_ratio"]["max"]*100:.2f}%
  中位数: {summary["culling_ratio"]["median"]*100:.2f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plot_path = os.path.join(scene_output_dir, "frustum_culling_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存到: {plot_path}")


def aggregate_dataset_results(base_dir, dataset_name, iteration, output_dir):
    """
    汇总整个数据集的所有场景结果
    
    Args:
        base_dir: 基础输出目录（如 output）
        dataset_name: 数据集名称（如 dynerf）
        iteration: 迭代次数
        output_dir: 汇总结果输出目录
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
            # 检查是否有point_cloud目录
            point_cloud_dir = os.path.join(scene_path, "point_cloud")
            if os.path.exists(point_cloud_dir):
                scenes.append(item)
    
    print(f"\n找到 {len(scenes)} 个场景: {scenes}")
    
    # 分析每个场景
    all_summaries = []
    for scene in scenes:
        model_path = os.path.join(dataset_dir, scene)
        # 尝试从cfg_args中获取source_path
        source_path = None
        cfg_path = os.path.join(model_path, "cfg_args")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r') as f:
                    cfg_content = f.read()
                    # 简单解析source_path
                    if 'source_path' in cfg_content:
                        import re
                        match = re.search(r"source_path['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", cfg_content)
                        if match:
                            source_path = match.group(1)
            except:
                pass
        
        if source_path is None:
            # 尝试从data目录推断
            data_dir = os.path.join("data", dataset_name, scene)
            if os.path.exists(data_dir):
                source_path = data_dir
            else:
                print(f"警告: 无法找到场景 {scene} 的source_path，跳过")
                continue
        
        try:
            summary = analyze_scene_frustum_culling(model_path, source_path, iteration)
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
    mean_culled = [s["culled_gaussians"]["mean"] for s in all_summaries]
    mean_visible = [s["visible_gaussians"]["mean"] for s in all_summaries]
    mean_culling_ratio = [s["culling_ratio"]["mean"] * 100 for s in all_summaries]  # 转换为百分比
    total_gaussians = [s["total_gaussians"] for s in all_summaries]
    
    # 创建汇总图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Frustum Culling Analysis - Dataset: {dataset_name}', fontsize=16, fontweight='bold')
    
    # 1. 平均被剔除的高斯球数量（按场景）
    x_pos = np.arange(len(scene_names))
    axes[0, 0].bar(x_pos, mean_culled, alpha=0.7, color='red')
    axes[0, 0].set_xlabel('Scene')
    axes[0, 0].set_ylabel('Mean Culled Gaussians')
    axes[0, 0].set_title('Mean Culled Gaussians per Scene')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(scene_names, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. 平均可见的高斯球数量（按场景）
    axes[0, 1].bar(x_pos, mean_visible, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Scene')
    axes[0, 1].set_ylabel('Mean Visible Gaussians')
    axes[0, 1].set_title('Mean Visible Gaussians per Scene')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(scene_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. 平均剔除比例（按场景）
    axes[1, 0].bar(x_pos, mean_culling_ratio, alpha=0.7, color='blue')
    axes[1, 0].set_xlabel('Scene')
    axes[1, 0].set_ylabel('Mean Culling Ratio (%)')
    axes[1, 0].set_title('Mean Culling Ratio per Scene')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(scene_names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. 总高斯球数量 vs 平均剔除数量（散点图）
    axes[1, 1].scatter(total_gaussians, mean_culled, s=100, alpha=0.6, color='purple')
    axes[1, 1].set_xlabel('Total Gaussians')
    axes[1, 1].set_ylabel('Mean Culled Gaussians')
    axes[1, 1].set_title('Total Gaussians vs Mean Culled')
    axes[1, 1].grid(True, alpha=0.3)
    # 添加场景名称标签
    for i, name in enumerate(scene_names):
        axes[1, 1].annotate(name, (total_gaussians[i], mean_culled[i]), 
                           fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plot_path = os.path.join(dataset_output_dir, f"{dataset_name.replace('/', '_')}_frustum_culling_aggregated.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"数据集汇总图表已保存到: {plot_path}")
    
    # 保存汇总统计
    aggregated_stats = {
        "dataset": dataset_name,
        "num_scenes": len(all_summaries),
        "scenes": {}
    }
    
    for summary in all_summaries:
        scene_name = summary["scene_name"]
        aggregated_stats["scenes"][scene_name] = {
            "total_gaussians": summary["total_gaussians"],
            "num_test_views": summary["num_test_views"],
            "mean_culled": summary["culled_gaussians"]["mean"],
            "mean_visible": summary["visible_gaussians"]["mean"],
            "mean_culling_ratio": summary["culling_ratio"]["mean"]
        }
    
    # 计算数据集级别的总体统计
    all_culled_means = [s["culled_gaussians"]["mean"] for s in all_summaries]
    all_visible_means = [s["visible_gaussians"]["mean"] for s in all_summaries]
    all_culling_ratios = [s["culling_ratio"]["mean"] for s in all_summaries]
    
    aggregated_stats["dataset_level"] = {
        "mean_culled_gaussians": float(np.mean(all_culled_means)),
        "std_culled_gaussians": float(np.std(all_culled_means)),
        "mean_visible_gaussians": float(np.mean(all_visible_means)),
        "std_visible_gaussians": float(np.std(all_visible_means)),
        "mean_culling_ratio": float(np.mean(all_culling_ratios)),
        "std_culling_ratio": float(np.std(all_culling_ratios))
    }
    
    json_path = os.path.join(dataset_output_dir, f"{dataset_name.replace('/', '_')}_frustum_culling_aggregated.json")
    with open(json_path, 'w') as f:
        json.dump(aggregated_stats, f, indent=2)
    
    print(f"数据集汇总统计已保存到: {json_path}")
    
    # 打印汇总信息
    print("\n" + "="*60)
    print(f"数据集汇总统计: {dataset_name}")
    print("="*60)
    print(f"场景数量: {len(all_summaries)}")
    print(f"\n平均被剔除的高斯球: {aggregated_stats['dataset_level']['mean_culled_gaussians']:.0f} ± {aggregated_stats['dataset_level']['std_culled_gaussians']:.0f}")
    print(f"平均可见的高斯球: {aggregated_stats['dataset_level']['mean_visible_gaussians']:.0f} ± {aggregated_stats['dataset_level']['std_visible_gaussians']:.0f}")
    print(f"平均剔除比例: {aggregated_stats['dataset_level']['mean_culling_ratio']*100:.2f}% ± {aggregated_stats['dataset_level']['std_culling_ratio']*100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    parser = ArgumentParser(description="Frustum Culling Analysis Script")
    parser.add_argument("--model_path", type=str, default="", help="单个场景的模型路径")
    parser.add_argument("--source_path", type=str, default="", help="单个场景的数据源路径")
    parser.add_argument("--dataset", type=str, default="", help="数据集名称（用于分析整个数据集）")
    parser.add_argument("--base_dir", type=str, default="output", help="基础输出目录")
    parser.add_argument("--iteration", type=int, default=-1, help="迭代次数")
    parser.add_argument("--output_dir", type=str, default="output", help="结果输出目录")
    
    args = parser.parse_args()
    
    # 初始化系统状态
    safe_state(False)
    
    if args.dataset:
        # 分析整个数据集
        output_dir = os.path.join(args.output_dir, args.dataset, "frustum_culling_analysis")
        os.makedirs(output_dir, exist_ok=True)
        aggregate_dataset_results(args.base_dir, args.dataset, args.iteration, output_dir)
    elif args.model_path and args.source_path:
        # 分析单个场景
        output_dir = os.path.join(args.output_dir, "frustum_culling_analysis")
        os.makedirs(output_dir, exist_ok=True)
        summary = analyze_scene_frustum_culling(args.model_path, args.source_path, args.iteration)
        save_scene_results(summary, output_dir)
        plot_scene_results(summary, output_dir)
    else:
        print("错误: 请提供 --model_path 和 --source_path（单个场景）或 --dataset（整个数据集）")
        parser.print_help()
