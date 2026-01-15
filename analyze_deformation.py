"""
Deformation Distribution Analysis - 高斯球形变量分布分析

对训练后模型的所有高斯球计算max_deformation，
生成统计信息和可视化图表，帮助确定lambda阈值的合理范围。

用法:
    python analyze_deformation.py -m output/dynerf/coffee_martini --configs arguments/dynerf/default.py
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from scene import Scene
from gaussian_renderer import GaussianModel
from scene.static_analyzer import StaticGaussianAnalyzer
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams


def analyze_deformation_distribution(
    dataset: ModelParams,
    hyperparam,
    iteration: int,
    num_time_samples: int = 50,
    output_dir: str = None,
):
    """
    分析高斯球形变量分布
    """
    print("=" * 60)
    print("Gaussian Deformation Distribution Analysis")
    print("=" * 60)
    
    with torch.no_grad():
        # 加载模型
        print("\n[1/3] Loading model...")
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        num_gaussians = gaussians._xyz.shape[0]
        print(f"   - Number of Gaussians: {num_gaussians}")
        print(f"   - Time samples: {num_time_samples}")
        
        # 分析形变量
        print("\n[2/3] Computing max deformation for each Gaussian...")
        analyzer = StaticGaussianAnalyzer(gaussians, num_time_samples=num_time_samples)
        max_deformation = analyzer.compute_max_deformation(verbose=True)
        
        # 转换为numpy
        max_def_np = max_deformation.cpu().numpy()
        
        # 计算统计信息
        stats = {
            'count': len(max_def_np),
            'min': float(np.min(max_def_np)),
            'max': float(np.max(max_def_np)),
            'mean': float(np.mean(max_def_np)),
            'std': float(np.std(max_def_np)),
            'median': float(np.median(max_def_np)),
            'percentiles': {
                '1%': float(np.percentile(max_def_np, 1)),
                '5%': float(np.percentile(max_def_np, 5)),
                '10%': float(np.percentile(max_def_np, 10)),
                '25%': float(np.percentile(max_def_np, 25)),
                '50%': float(np.percentile(max_def_np, 50)),
                '75%': float(np.percentile(max_def_np, 75)),
                '90%': float(np.percentile(max_def_np, 90)),
                '95%': float(np.percentile(max_def_np, 95)),
                '99%': float(np.percentile(max_def_np, 99)),
                '99.9%': float(np.percentile(max_def_np, 99.9)),
            }
        }
        
        # 打印统计信息
        print("\n" + "=" * 60)
        print("Max Deformation Statistics")
        print("=" * 60)
        print(f"  Total Gaussians: {stats['count']:,}")
        print(f"  Min:    {stats['min']:.8f}")
        print(f"  Max:    {stats['max']:.8f}")
        print(f"  Mean:   {stats['mean']:.8f}")
        print(f"  Std:    {stats['std']:.8f}")
        print(f"  Median: {stats['median']:.8f}")
        print("\n  Percentiles:")
        for k, v in stats['percentiles'].items():
            # 计算该百分位对应的静态高斯球数量
            pct = float(k.replace('%', ''))
            count = int(pct / 100 * stats['count'])
            print(f"    {k:>6}: {v:.8f}  ({count:,} Gaussians below)")
        
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.join(dataset.model_path, "deformation_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存统计信息
        stats_path = os.path.join(output_dir, "deformation_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n  Statistics saved to {stats_path}")
        
        # 保存原始数据
        data_path = os.path.join(output_dir, "max_deformation.npy")
        np.save(data_path, max_def_np)
        print(f"  Raw data saved to {data_path}")
        
        # 生成可视化
        print("\n[3/3] Generating visualizations...")
        plot_deformation_distribution(max_def_np, stats, output_dir)
        
        # 建议的lambda值
        print("\n" + "=" * 60)
        print("Suggested Lambda Values for Sensitivity Analysis")
        print("=" * 60)
        suggested_lambdas = suggest_lambda_values(stats)
        for desc, val in suggested_lambdas:
            print(f"  {desc}: {val:.8f}")
        
        return max_def_np, stats


def plot_deformation_distribution(max_def_np, stats, output_dir):
    """生成形变量分布可视化图表"""
    
    # 设置中文字体（如果需要）
    plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 直方图 (线性尺度)
    ax1 = fig.add_subplot(2, 3, 1)
    counts, bins, _ = ax1.hist(max_def_np, bins=100, color='#3498db', alpha=0.7, edgecolor='white')
    ax1.axvline(stats['mean'], color='#e74c3c', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.6f}")
    ax1.axvline(stats['median'], color='#2ecc71', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.6f}")
    ax1.set_xlabel('Max Deformation', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Distribution (Linear Scale)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. 直方图 (对数Y轴)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(max_def_np, bins=100, color='#3498db', alpha=0.7, edgecolor='white')
    ax2.set_yscale('log')
    ax2.axvline(stats['mean'], color='#e74c3c', linestyle='--', linewidth=2, label=f"Mean")
    ax2.axvline(stats['median'], color='#2ecc71', linestyle='--', linewidth=2, label=f"Median")
    ax2.set_xlabel('Max Deformation', fontsize=11)
    ax2.set_ylabel('Count (log)', fontsize=11)
    ax2.set_title('Distribution (Log Y-axis)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. 直方图 (对数X轴，排除零值)
    ax3 = fig.add_subplot(2, 3, 3)
    non_zero = max_def_np[max_def_np > 0]
    if len(non_zero) > 0:
        log_bins = np.geomspace(non_zero.min(), non_zero.max(), 100)
        ax3.hist(non_zero, bins=log_bins, color='#9b59b6', alpha=0.7, edgecolor='white')
        ax3.set_xscale('log')
        ax3.axvline(stats['mean'], color='#e74c3c', linestyle='--', linewidth=2, label=f"Mean")
        ax3.axvline(stats['median'], color='#2ecc71', linestyle='--', linewidth=2, label=f"Median")
    ax3.set_xlabel('Max Deformation (log)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Distribution (Log X-axis)', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. 累积分布函数 (CDF)
    ax4 = fig.add_subplot(2, 3, 4)
    sorted_data = np.sort(max_def_np)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
    ax4.plot(sorted_data, cdf, color='#e67e22', linewidth=2)
    
    # 标记关键百分位
    percentiles_to_mark = [10, 25, 50, 75, 90, 95, 99]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(percentiles_to_mark)))
    for pct, color in zip(percentiles_to_mark, colors):
        val = np.percentile(max_def_np, pct)
        ax4.axhline(pct, color=color, linestyle=':', alpha=0.5)
        ax4.axvline(val, color=color, linestyle=':', alpha=0.5)
        ax4.plot(val, pct, 'o', color=color, markersize=8)
        ax4.annotate(f'{pct}%: {val:.6f}', (val, pct), 
                    xytext=(10, 0), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Max Deformation', fontsize=11)
    ax4.set_ylabel('Cumulative % of Gaussians', fontsize=11)
    ax4.set_title('Cumulative Distribution Function (CDF)', fontsize=12)
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)
    
    # 5. CDF (对数X轴)
    ax5 = fig.add_subplot(2, 3, 5)
    if len(non_zero) > 0:
        sorted_nz = np.sort(non_zero)
        cdf_nz = np.arange(1, len(sorted_nz) + 1) / len(max_def_np) * 100
        ax5.plot(sorted_nz, cdf_nz, color='#e67e22', linewidth=2)
        ax5.set_xscale('log')
        
        for pct, color in zip(percentiles_to_mark, colors):
            val = np.percentile(max_def_np, pct)
            if val > 0:
                ax5.axhline(pct, color=color, linestyle=':', alpha=0.5)
                ax5.axvline(val, color=color, linestyle=':', alpha=0.5)
                ax5.plot(val, pct, 'o', color=color, markersize=8)
    
    ax5.set_xlabel('Max Deformation (log)', fontsize=11)
    ax5.set_ylabel('Cumulative % of Gaussians', fontsize=11)
    ax5.set_title('CDF (Log X-axis)', fontsize=12)
    ax5.set_ylim(0, 100)
    ax5.grid(True, alpha=0.3)
    
    # 6. 百分位数条形图
    ax6 = fig.add_subplot(2, 3, 6)
    percentile_names = list(stats['percentiles'].keys())
    percentile_values = list(stats['percentiles'].values())
    y_pos = np.arange(len(percentile_names))
    
    bars = ax6.barh(y_pos, percentile_values, color='#1abc9c', alpha=0.8)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(percentile_names)
    ax6.set_xlabel('Max Deformation', fontsize=11)
    ax6.set_title('Percentile Values', fontsize=12)
    ax6.grid(True, alpha=0.3, axis='x')
    
    # 在条形上添加数值标签
    for bar, val in zip(bars, percentile_values):
        ax6.text(val + max(percentile_values) * 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.6f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存主图
    main_plot_path = os.path.join(output_dir, "deformation_distribution.png")
    plt.savefig(main_plot_path, dpi=150, bbox_inches='tight')
    print(f"  Main plot saved to {main_plot_path}")
    
    plt.close()
    
    # 生成简化的单独CDF图（用于论文/报告）
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_data, cdf, color='#2980b9', linewidth=2.5)
    
    # 标记关键点
    key_percentiles = [50, 90, 95, 99]
    for pct in key_percentiles:
        val = np.percentile(max_def_np, pct)
        ax.axhline(pct, color='#bdc3c7', linestyle='--', alpha=0.7)
        ax.plot(val, pct, 'o', color='#e74c3c', markersize=10, zorder=5)
        ax.annotate(f'{pct}%\nλ={val:.4f}', (val, pct), 
                   xytext=(15, -5), textcoords='offset points', fontsize=10,
                   fontweight='bold')
    
    ax.set_xlabel('Lambda Threshold (λ = Max Deformation)', fontsize=12)
    ax.set_ylabel('Static Gaussians (%)', fontsize=12)
    ax.set_title('Percentage of Static Gaussians vs Lambda Threshold', fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, stats['percentiles']['99.9%'] * 1.1)
    ax.grid(True, alpha=0.3)
    
    # 添加注释
    ax.text(0.98, 0.02, f"Total Gaussians: {stats['count']:,}\nMean: {stats['mean']:.6f}\nMedian: {stats['median']:.6f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    cdf_plot_path = os.path.join(output_dir, "static_ratio_vs_lambda.png")
    plt.savefig(cdf_plot_path, dpi=150, bbox_inches='tight')
    print(f"  CDF plot saved to {cdf_plot_path}")
    
    plt.close()
    
    # 生成箱线图
    fig3, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(max_def_np, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][0].set_alpha(0.7)
    
    ax.set_ylabel('Max Deformation', fontsize=12)
    ax.set_title('Max Deformation Box Plot', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加统计文本
    textstr = f"Min: {stats['min']:.6f}\nQ1: {stats['percentiles']['25%']:.6f}\nMedian: {stats['median']:.6f}\nQ3: {stats['percentiles']['75%']:.6f}\nMax: {stats['max']:.6f}"
    ax.text(1.15, 0.5, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    box_plot_path = os.path.join(output_dir, "deformation_boxplot.png")
    plt.savefig(box_plot_path, dpi=150, bbox_inches='tight')
    print(f"  Box plot saved to {box_plot_path}")
    
    plt.close()


def suggest_lambda_values(stats):
    """基于统计信息建议lambda值"""
    suggestions = [
        ("Very Conservative (1% static)", stats['percentiles']['1%']),
        ("Conservative (10% static)", stats['percentiles']['10%']),
        ("Moderate (25% static)", stats['percentiles']['25%']),
        ("Balanced (50% static / Median)", stats['percentiles']['50%']),
        ("Aggressive (75% static)", stats['percentiles']['75%']),
        ("Very Aggressive (90% static)", stats['percentiles']['90%']),
        ("Extreme (95% static)", stats['percentiles']['95%']),
        ("Maximum (99% static)", stats['percentiles']['99%']),
    ]
    return suggestions


if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze Gaussian Deformation Distribution")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int,
                        help="Model iteration to load (-1 for latest)")
    parser.add_argument("--configs", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--num_time_samples", type=int, default=50,
                        help="Number of time samples for deformation analysis")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    
    args = get_combined_args(parser)
    print("Analyzing:", args.model_path)
    
    # Save custom args before merge (merge_hparams may overwrite them)
    output_dir = getattr(args, 'output_dir', None)
    num_time_samples = getattr(args, 'num_time_samples', 50)
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    safe_state(False)
    
    analyze_deformation_distribution(
        model.extract(args),
        hyperparam.extract(args),
        args.iteration,
        num_time_samples=num_time_samples,
        output_dir=output_dir,
    )
