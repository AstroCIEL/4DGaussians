"""
Aggregate Sensitivity Analysis Results
汇总所有场景的敏感性分析结果，生成数据集级别的汇总可视化

用法:
    python scripts/aggregate_sensitivity_results.py --dataset dynerf --output_dir output/dynerf/aggregated_analysis
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict


def load_scene_results(result_path, scene_name):
    """加载单个场景的结果"""
    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Failed to load {result_path}: {e}")
        return None


def extract_metric_deterioration(results_data, metric_name, standard_ratios=None):
    """
    从结果中提取指标恶化值
    
    Args:
        results_data: 场景的sensitivity_results.json数据
        metric_name: 'psnr', 'ssim', 'lpips'
        standard_ratios: 标准比例列表（用于匹配lambda模式的结果）
    
    Returns:
        dict: {static_ratio: deterioration_value}
    """
    deterioration = {}
    baseline = results_data['baseline_metrics'].get(metric_name, {}).get('mean')
    
    if baseline is None:
        return deterioration
    
    for result in results_data['results']:
        # 优先使用target_static_ratio（ratio模式），否则使用static_ratio（lambda模式）
        target_ratio = result.get('target_static_ratio')
        if target_ratio is None:
            # 对于lambda模式，使用static_ratio匹配到最接近的标准比例
            actual_ratio = result.get('static_ratio')
            if actual_ratio is None or standard_ratios is None:
                continue
            # 找到最接近的标准比例
            target_ratio = min(standard_ratios, key=lambda x: abs(x - actual_ratio))
            # 只接受误差小于2%的匹配
            if abs(target_ratio - actual_ratio) > 0.02:
                continue
        else:
            # 确保ratio模式的target_ratio也在标准列表中（如果有的话）
            if standard_ratios is not None and target_ratio not in standard_ratios:
                # 匹配到最接近的标准比例
                target_ratio = min(standard_ratios, key=lambda x: abs(x - target_ratio))
        
        if metric_name == 'psnr':
            # PSNR已经有drop值
            drop = result.get('psnr_drop', 0)
            deterioration[target_ratio] = drop
        elif metric_name == 'ssim':
            # SSIM: baseline - current (越大越好，所以drop是恶化)
            current = result.get('ssim_mean')
            if current is not None:
                deterioration[target_ratio] = baseline - current
        elif metric_name == 'lpips':
            # LPIPS: current - baseline (越小越好，所以increase是恶化)
            current = result.get('lpips_mean')
            if current is not None:
                deterioration[target_ratio] = current - baseline
    
    return deterioration


def aggregate_scenes(base_dir, dataset_name, scenes):
    """
    汇总所有场景的结果
    
    Returns:
        dict: {
            'psnr': {scene: {ratio: drop}},
            'ssim': {scene: {ratio: drop}},
            'lpips': {scene: {ratio: increase}}
        }
    """
    aggregated = defaultdict(lambda: defaultdict(dict))
    
    # 首先收集所有可能的静态比例（从target_static_ratios字段）
    all_ratios_set = set()
    for scene in scenes:
        result_path = os.path.join(base_dir, dataset_name, scene, "sensitivity_analysis", "sensitivity_results.json")
        if os.path.exists(result_path):
            data = load_scene_results(result_path, scene)
            if data is not None:
                target_ratios = data.get('target_static_ratios', [])
                # 转换为小数形式（如果是以百分比形式存储）
                for r in target_ratios:
                    if isinstance(r, (int, float)):
                        ratio = r / 100.0 if r > 1 else r
                        all_ratios_set.add(ratio)
    
    # 如果没有找到标准比例，使用常见的5%步长
    if not all_ratios_set:
        all_ratios_set = {r/100.0 for r in range(5, 100, 5)}
    
    standard_ratios = sorted(all_ratios_set)
    
    for scene in scenes:
        result_path = os.path.join(base_dir, dataset_name, scene, "sensitivity_analysis", "sensitivity_results.json")
        
        if not os.path.exists(result_path):
            print(f"Warning: {result_path} not found, skipping scene {scene}")
            continue
        
        data = load_scene_results(result_path, scene)
        if data is None:
            continue
        
        metrics = data.get('metrics', [])
        
        for metric in metrics:
            deterioration = extract_metric_deterioration(data, metric, standard_ratios)
            aggregated[metric][scene] = deterioration
    
    return aggregated


def create_summary_table(aggregated_data, output_path):
    """创建汇总表格（CSV）"""
    all_metrics = ['psnr', 'ssim', 'lpips']
    all_scenes = set()
    all_ratios = set()
    
    # 收集所有场景和比例
    for metric_data in aggregated_data.values():
        all_scenes.update(metric_data.keys())
        for scene_data in metric_data.values():
            all_ratios.update(scene_data.keys())
    
    all_scenes = sorted(all_scenes)
    all_ratios = sorted(all_ratios)
    
    # 为每个指标创建表格
    for metric in all_metrics:
        if metric not in aggregated_data:
            continue
        
        rows = []
        for scene in all_scenes:
            row = {'Scene': scene}
            scene_data = aggregated_data[metric].get(scene, {})
            for ratio in all_ratios:
                row[f'{ratio*100:.0f}%'] = scene_data.get(ratio, np.nan)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_path, f'{metric}_summary_table.csv')
        df.to_csv(csv_path, index=False)
        print(f"   Summary table saved: {csv_path}")
    
    return all_scenes, all_ratios


def plot_aggregated_results(aggregated_data, output_dir, scenes, ratios,dataset_name="dynerf"):
    """生成汇总可视化图表"""
    metrics_info = {
        'psnr': {'label': 'PSNR Drop (dB)', 'title': 'PSNR Deterioration', 'color': '#2E86AB'},
        'ssim': {'label': 'SSIM Drop', 'title': 'SSIM Deterioration', 'color': '#28A745'},
        'lpips': {'label': 'LPIPS Increase', 'title': 'LPIPS Deterioration', 'color': '#E94F37'},
    }
    
    # 为每个指标创建图表
    for metric, info in metrics_info.items():
        if metric not in aggregated_data:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metric_data = aggregated_data[metric]
        colors = plt.cm.tab10(np.linspace(0, 1, len(scenes)))
        
        # 绘制每个场景的曲线
        for idx, scene in enumerate(scenes):
            if scene not in metric_data:
                continue
            
            scene_data = metric_data[scene]
            x_vals = []
            y_vals = []
            
            for ratio in ratios:
                if ratio in scene_data:
                    x_vals.append(ratio * 100)  # 转换为百分比
                    y_vals.append(scene_data[ratio])
            
            if len(x_vals) > 0:
                ax.plot(x_vals, y_vals, 'o-', label=scene, color=colors[idx], 
                       linewidth=2, markersize=5, alpha=0.8)
        
        ax.set_xlabel('Static Gaussians Ratio (%)', fontsize=12)
        ax.set_ylabel(info['label'], fontsize=12)
        ax.set_title(f'{info["title"]} Across Scenes ({dataset_name} Dataset)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{metric}_aggregated.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   Aggregated plot saved: {plot_path}")
        plt.close(fig)
    
    # 创建热力图（每个指标一个）
    for metric, info in metrics_info.items():
        if metric not in aggregated_data:
            continue
        
        metric_data = aggregated_data[metric]
        
        # 构建矩阵: rows=scenes, cols=ratios
        matrix = []
        for scene in scenes:
            row = []
            scene_data = metric_data.get(scene, {})
            for ratio in ratios:
                row.append(scene_data.get(ratio, np.nan))
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(14, 6))
        
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        # 设置标签
        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels([f'{r*100:.0f}%' for r in ratios])
        ax.set_yticks(range(len(scenes)))
        ax.set_yticklabels(scenes)
        
        # 添加数值标注
        for i in range(len(scenes)):
            for j in range(len(ratios)):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=7)
        
        ax.set_xlabel('Static Gaussians Ratio (%)', fontsize=12)
        ax.set_ylabel('Scene', fontsize=12)
        ax.set_title(f'{info["title"]} Heatmap ({dataset_name} Dataset)', fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(info['label'], fontsize=11)
        
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, f'{metric}_heatmap.png')
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        print(f"   Heatmap saved: {heatmap_path}")
        plt.close(fig)


def plot_statistics_summary(aggregated_data, output_dir, scenes, ratios,dataset_name="dynerf"):
    """生成统计汇总图（均值、标准差等）"""
    metrics_info = {
        'psnr': {'label': 'PSNR Drop (dB)', 'title': 'PSNR Deterioration Statistics'},
        'ssim': {'label': 'SSIM Drop', 'title': 'SSIM Deterioration Statistics'},
        'lpips': {'label': 'LPIPS Increase', 'title': 'LPIPS Deterioration Statistics'},
    }
    
    for metric, info in metrics_info.items():
        if metric not in aggregated_data:
            continue
        
        metric_data = aggregated_data[metric]
        
        # 计算每个比例的统计信息
        means = []
        stds = []
        medians = []
        ratios_list = []
        
        for ratio in ratios:
            values = []
            for scene in scenes:
                if scene in metric_data and ratio in metric_data[scene]:
                    values.append(metric_data[scene][ratio])
            
            if len(values) > 0:
                means.append(np.mean(values))
                stds.append(np.std(values))
                medians.append(np.median(values))
                ratios_list.append(ratio * 100)
        
        if len(means) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ratios_list = np.array(ratios_list)
        means = np.array(means)
        stds = np.array(stds)
        medians = np.array(medians)
        
        # 绘制均值和误差条
        ax.errorbar(ratios_list, means, yerr=stds, fmt='o-', capsize=5,
                   color='#2E86AB', linewidth=2, markersize=6, label='Mean ± Std')
        ax.plot(ratios_list, medians, 's--', color='#E94F37', linewidth=2,
               markersize=6, label='Median', alpha=0.8)
        
        ax.set_xlabel('Static Gaussians Ratio (%)', fontsize=12)
        ax.set_ylabel(info['label'], fontsize=12)
        ax.set_title(f'{info["title"]} Across All Scenes ({dataset_name} Dataset)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        stats_path = os.path.join(output_dir, f'{metric}_statistics.png')
        plt.savefig(stats_path, dpi=150, bbox_inches='tight')
        print(f"   Statistics plot saved: {stats_path}")
        plt.close(fig)


def main():
    parser = ArgumentParser(description="Aggregate Sensitivity Analysis Results")
    parser.add_argument("--dataset", type=str, default="dynerf",
                       help="Dataset name")
    parser.add_argument("--base_dir", type=str, default="output",
                       help="Base directory containing results")
    parser.add_argument("--scenes", type=str, nargs='+',
                       help="List of scenes to aggregate (default: auto-detect)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for aggregated results")
    
    args = parser.parse_args()
    
    # 确定场景列表
    if args.scenes:
        scenes = args.scenes
    else:
        # 自动检测场景
        dataset_dir = os.path.join(args.base_dir, args.dataset)
        if os.path.exists(dataset_dir):
            scenes = [d for d in os.listdir(dataset_dir)
                     if os.path.isdir(os.path.join(dataset_dir, d))
                     and os.path.exists(os.path.join(dataset_dir, d, "sensitivity_analysis", "sensitivity_results.json"))]
            scenes = sorted(scenes)
        else:
            print(f"Error: Dataset directory {dataset_dir} not found")
            return
    
    print("=" * 60)
    print("Aggregating Sensitivity Analysis Results")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Scenes: {', '.join(scenes)}")
    print("=" * 60)
    
    # 汇总数据
    aggregated = aggregate_scenes(args.base_dir, args.dataset, scenes)
    
    if not aggregated:
        print("Error: No valid results found")
        return
    
    # 确定输出目录
    if args.output_dir is None:
        output_dir = os.path.join(args.base_dir, args.dataset, "aggregated_analysis")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有比例（从第一个场景的结果中推断）
    all_ratios = set()
    for metric_data in aggregated.values():
        for scene_data in metric_data.values():
            all_ratios.update(scene_data.keys())
    all_ratios = sorted(all_ratios)
    
    print(f"\nFound static ratios: {[f'{r*100:.0f}%' for r in all_ratios]}")
    print(f"Found metrics: {list(aggregated.keys())}")
    
    # 创建汇总表格
    print("\n[1/3] Creating summary tables...")
    create_summary_table(aggregated, output_dir)
    
    # 创建可视化图表
    print("\n[2/3] Creating aggregated plots...")
    plot_aggregated_results(aggregated, output_dir, scenes, all_ratios,args.dataset)
    
    # 创建统计汇总
    print("\n[3/3] Creating statistics summary...")
    plot_statistics_summary(aggregated, output_dir, scenes, all_ratios,args.dataset)
    
    # 保存汇总数据
    summary_data = {
        'dataset': args.dataset,
        'scenes': scenes,
        'ratios': all_ratios,
        'aggregated_results': {
            metric: {
                scene: {
                    str(ratio): value for ratio, value in scene_data.items()
                } for scene, scene_data in metric_data.items()
            } for metric, metric_data in aggregated.items()
        }
    }
    
    summary_json_path = os.path.join(output_dir, "aggregated_results.json")
    with open(summary_json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n   Summary JSON saved: {summary_json_path}")
    
    print("\n" + "=" * 60)
    print("Aggregation complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
