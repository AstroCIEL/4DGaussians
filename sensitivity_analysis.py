"""
Static Gaussian Sensitivity Analysis - 静止高斯球阈值敏感性分析

对不同的lambda阈值进行测试,计算渲染质量(PSNR),
生成PSNR-lambda曲线图和静态比例-lambda曲线图。

用法:
    python sensitivity_analysis.py -m output/dynerf/coffee_martini --configs arguments/dynerf/default.py
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path

from scene import Scene
from gaussian_renderer import render, GaussianModel
from scene.static_analyzer import StaticGaussianAnalyzer
from utils.general_utils import safe_state
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams


def compute_psnr_for_views(views, gaussians, pipeline, background, cam_type):
    """计算给定视图集的平均PSNR"""
    psnrs = []
    for view in views:
        rendering_results = render(view, gaussians, pipeline, background, cam_type=cam_type)
        rendering = rendering_results["render"]
        
        if cam_type != "PanopticSports":
            gt = view.original_image[0:3, :, :].cuda()
        else:
            gt = view['image'].cuda()
        
        psnr_val = psnr(rendering.unsqueeze(0), gt.unsqueeze(0))
        psnrs.append(psnr_val.item())
    
    return np.mean(psnrs), np.std(psnrs), psnrs


def run_sensitivity_analysis(
    dataset: ModelParams,
    hyperparam,
    iteration: int,
    pipeline: PipelineParams,
    lambda_values: list = None,
    num_time_samples: int = 50,
    output_dir: str = None,
):
    """
    运行静止高斯球阈值敏感性分析
    
    Args:
        dataset: 模型参数
        hyperparam: 隐藏参数
        iteration: 加载的迭代次数
        pipeline: 管道参数
        lambda_values: 要测试的lambda阈值列表
        num_time_samples: 时间采样点数量
        output_dir: 输出目录
    """
    print("=" * 60)
    print("Static Gaussian Sensitivity Analysis")
    print("=" * 60)
    
    with torch.no_grad():
        # 加载模型
        print("\n[1/4] Loading model...")
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type = scene.dataset_type
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 获取测试视图
        test_views = scene.getTestCameras()
        if len(test_views) == 0:
            print("Warning: No test views found, using train views")
            test_views = scene.getTrainCameras()
        
        print(f"   - Number of Gaussians: {gaussians._xyz.shape[0]}")
        print(f"   - Number of test views: {len(test_views)}")
        
        # 分析静止性
        print("\n[2/4] Analyzing gaussian deformation...")
        analyzer = StaticGaussianAnalyzer(gaussians, num_time_samples=num_time_samples)
        max_deformation = analyzer.compute_max_deformation(verbose=True)
        
        stats = analyzer.get_statistics()
        print(f"\n   Deformation Statistics:")
        print(f"   - Min:    {stats['min']:.6f}")
        print(f"   - Max:    {stats['max']:.6f}")
        print(f"   - Mean:   {stats['mean']:.6f}")
        print(f"   - Median: {stats['median']:.6f}")
        print(f"   - Std:    {stats['std']:.6f}")
        
        # 确定lambda值范围
        if lambda_values is None:
            _, _, suggested = analyzer.suggest_lambda_range()
            # 添加0和一些额外的点以获得更完整的曲线
            lambda_values = [0.0] + list(suggested) + [stats['max'] * 1.5]
            lambda_values = sorted(set(lambda_values))
        
        print(f"\n   Testing {len(lambda_values)} lambda values")
        print(f"   Range: [{min(lambda_values):.6f}, {max(lambda_values):.6f}]")
        
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.join(dataset.model_path, "sensitivity_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存分析结果
        analysis_path = os.path.join(output_dir, "deformation_analysis.pth")
        analyzer.save_analysis(analysis_path)
        
        # 首先计算baseline (无优化)
        print("\n[3/4] Computing baseline PSNR (no optimization)...")
        gaussians._deformation.clear_static_mask()
        baseline_psnr, baseline_std, _ = compute_psnr_for_views(
            test_views, gaussians, pipeline, background, cam_type
        )
        print(f"   Baseline PSNR: {baseline_psnr:.4f} ± {baseline_std:.4f}")
        
        # 对每个lambda值进行测试
        print("\n[4/4] Running sensitivity analysis...")
        results = []
        
        for lambda_val in tqdm(lambda_values, desc="Testing lambda values"):
            # 获取静止掩码
            static_mask = analyzer.get_static_mask(lambda_val)
            static_ratio = static_mask.float().mean().item()
            
            # 设置静止掩码
            gaussians._deformation.set_static_mask(static_mask)
            
            # 计算PSNR
            mean_psnr, std_psnr, per_view_psnrs = compute_psnr_for_views(
                test_views, gaussians, pipeline, background, cam_type
            )
            
            results.append({
                'lambda': lambda_val,
                'psnr_mean': mean_psnr,
                'psnr_std': std_psnr,
                'static_ratio': static_ratio,
                'num_static': static_mask.sum().item(),
                'num_total': static_mask.shape[0],
                'psnr_drop': baseline_psnr - mean_psnr,
                'per_view_psnrs': per_view_psnrs,
            })
        
        # 清除静止掩码
        gaussians._deformation.clear_static_mask()
        
        # 保存结果
        results_data = {
            'baseline_psnr': baseline_psnr,
            'baseline_std': baseline_std,
            'deformation_stats': stats,
            'results': [{k: v for k, v in r.items() if k != 'per_view_psnrs'} for r in results],
            'model_path': dataset.model_path,
            'iteration': iteration,
            'num_time_samples': num_time_samples,
        }
        
        results_path = os.path.join(output_dir, "sensitivity_results.json")
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\n   Results saved to {results_path}")
        
        # 生成图表
        plot_results(results, baseline_psnr, output_dir)
        
        return results, baseline_psnr


def plot_results(results, baseline_psnr, output_dir):
    """生成PSNR-lambda曲线图"""
    
    lambdas = [r['lambda'] for r in results]
    psnrs = [r['psnr_mean'] for r in results]
    psnr_stds = [r['psnr_std'] for r in results]
    static_ratios = [r['static_ratio'] * 100 for r in results]
    psnr_drops = [r['psnr_drop'] for r in results]
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Static Gaussian Threshold Sensitivity Analysis', fontsize=14, fontweight='bold')
    
    # 1. PSNR vs Lambda
    ax1 = axes[0, 0]
    ax1.errorbar(lambdas, psnrs, yerr=psnr_stds, fmt='o-', capsize=3, 
                 color='#2E86AB', linewidth=2, markersize=4, label='PSNR')
    ax1.axhline(y=baseline_psnr, color='#E94F37', linestyle='--', 
                linewidth=2, label=f'Baseline ({baseline_psnr:.2f})')
    ax1.set_xlabel('Lambda Threshold (λ)', fontsize=11)
    ax1.set_ylabel('PSNR (dB)', fontsize=11)
    ax1.set_title('Rendering Quality vs Threshold', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('symlog', linthresh=1e-4)
    
    # 2. Static Ratio vs Lambda
    ax2 = axes[0, 1]
    ax2.plot(lambdas, static_ratios, 'o-', color='#28A745', linewidth=2, markersize=4)
    ax2.fill_between(lambdas, 0, static_ratios, alpha=0.3, color='#28A745')
    ax2.set_xlabel('Lambda Threshold (λ)', fontsize=11)
    ax2.set_ylabel('Static Gaussians (%)', fontsize=11)
    ax2.set_title('Static Ratio vs Threshold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('symlog', linthresh=1e-4)
    ax2.set_ylim(0, 100)
    
    # 3. PSNR Drop vs Lambda
    ax3 = axes[1, 0]
    colors = ['#28A745' if d < 0.1 else '#FFC107' if d < 0.5 else '#E94F37' for d in psnr_drops]
    ax3.bar(range(len(lambdas)), psnr_drops, color=colors, alpha=0.8)
    ax3.axhline(y=0.1, color='#FFC107', linestyle='--', linewidth=1.5, label='0.1 dB threshold')
    ax3.axhline(y=0.5, color='#E94F37', linestyle='--', linewidth=1.5, label='0.5 dB threshold')
    ax3.set_xlabel('Lambda Index', fontsize=11)
    ax3.set_ylabel('PSNR Drop (dB)', fontsize=11)
    ax3.set_title('Quality Degradation vs Threshold', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Trade-off: PSNR Drop vs Static Ratio
    ax4 = axes[1, 1]
    scatter = ax4.scatter(static_ratios, psnr_drops, c=lambdas, cmap='viridis', 
                          s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Lambda (λ)', fontsize=10)
    ax4.set_xlabel('Static Gaussians (%)', fontsize=11)
    ax4.set_ylabel('PSNR Drop (dB)', fontsize=11)
    ax4.set_title('Quality-Efficiency Trade-off', fontsize=12)
    ax4.axhline(y=0.1, color='#FFC107', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.axhline(y=0.5, color='#E94F37', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.grid(True, alpha=0.3)
    
    # 在trade-off图上标注一些关键点
    for i, (sr, pd, lam) in enumerate(zip(static_ratios, psnr_drops, lambdas)):
        if i % max(1, len(lambdas) // 5) == 0:  # 只标注部分点
            ax4.annotate(f'λ={lam:.1e}', (sr, pd), fontsize=7, 
                        xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, "sensitivity_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   Plot saved to {plot_path}")
    
    # 额外保存一个简单的PSNR-lambda曲线
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(lambdas, psnrs, yerr=psnr_stds, fmt='o-', capsize=3,
                color='#2E86AB', linewidth=2, markersize=6, label='PSNR with optimization')
    ax.axhline(y=baseline_psnr, color='#E94F37', linestyle='--',
               linewidth=2, label=f'Baseline PSNR ({baseline_psnr:.2f} dB)')
    
    # 添加静态比例作为次坐标轴
    ax2 = ax.twinx()
    ax2.plot(lambdas, static_ratios, 's--', color='#28A745', linewidth=1.5, 
             markersize=5, alpha=0.7, label='Static Ratio')
    ax2.set_ylabel('Static Gaussians (%)', fontsize=11, color='#28A745')
    ax2.tick_params(axis='y', labelcolor='#28A745')
    ax2.set_ylim(0, 100)
    
    ax.set_xlabel('Lambda Threshold (λ)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('PSNR vs Lambda Threshold (Static Gaussian Optimization)', fontsize=13)
    ax.set_xscale('symlog', linthresh=1e-4)
    ax.grid(True, alpha=0.3)
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    simple_plot_path = os.path.join(output_dir, "psnr_lambda_curve.png")
    plt.savefig(simple_plot_path, dpi=150, bbox_inches='tight')
    print(f"   Simple plot saved to {simple_plot_path}")
    
    plt.close('all')
    
    # 打印推荐的lambda值
    print("\n" + "=" * 60)
    print("Recommended Lambda Values:")
    print("=" * 60)
    
    # 找到PSNR下降小于0.1dB的最大静态比例
    for threshold in [0.05, 0.1, 0.2, 0.5]:
        valid_results = [r for r in results if r['psnr_drop'] < threshold]
        if valid_results:
            best = max(valid_results, key=lambda x: x['static_ratio'])
            print(f"   PSNR drop < {threshold} dB: λ = {best['lambda']:.6f} "
                  f"(static: {best['static_ratio']*100:.1f}%, PSNR: {best['psnr_mean']:.2f})")


if __name__ == "__main__":
    parser = ArgumentParser(description="Static Gaussian Sensitivity Analysis")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int,
                        help="Model iteration to load (-1 for latest)")
    parser.add_argument("--configs", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--num_lambdas", type=int, default=25,
                        help="Number of lambda values to test")
    parser.add_argument("--num_time_samples", type=int, default=50,
                        help="Number of time samples for deformation analysis")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--lambda_min", type=float, default=None,
                        help="Minimum lambda value")
    parser.add_argument("--lambda_max", type=float, default=None,
                        help="Maximum lambda value")
    
    args = get_combined_args(parser)
    print("Analyzing:", args.model_path)
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    safe_state(False)
    
    # 确定lambda值范围
    lambda_values = None
    if args.lambda_min is not None and args.lambda_max is not None:
        lambda_values = np.geomspace(args.lambda_min, args.lambda_max, args.num_lambdas).tolist()
        lambda_values = [0.0] + lambda_values  # 始终包含0
    
    run_sensitivity_analysis(
        model.extract(args),
        hyperparam.extract(args),
        args.iteration,
        pipeline.extract(args),
        lambda_values=lambda_values,
        num_time_samples=args.num_time_samples,
        output_dir=args.output_dir,
    )
