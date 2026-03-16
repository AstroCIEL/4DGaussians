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

# 添加项目根目录到Python路径，确保可以导入模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scene import Scene
from gaussian_renderer import render, GaussianModel
from scene.static_analyzer import StaticGaussianAnalyzer
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams


def compute_metrics_for_views(views, gaussians, pipeline, background, cam_type, metrics, lpips_model=None, time_segment_masks=None):
    """
    计算给定视图集的平均指标
    
    Args:
        views: 视图列表
        gaussians: 高斯模型
        pipeline: 渲染管道
        background: 背景颜色
        cam_type: 相机类型
        metrics: 指标列表
        lpips_model: LPIPS模型（可选）
        time_segment_masks: 时间段mask字典，格式为 {(t_start, t_end): static_mask}
    """
    metrics = [m.lower() for m in metrics]
    values = {m: [] for m in metrics}
    
    for view in views:
        # 如果使用时间段模式，根据视图时间设置对应的static mask
        if time_segment_masks is not None:
            view_time = getattr(view, 'time', None)
            if view_time is not None:
                # 找到视图时间所在的时间段
                mask_set = False
                for (t_start, t_end), static_mask in time_segment_masks.items():
                    if t_start <= view_time <= t_end:
                        gaussians._deformation.set_static_mask(static_mask)
                        mask_set = True
                        break
                if not mask_set:
                    # 如果没找到对应时间段，清除mask
                    gaussians._deformation.clear_static_mask()
            else:
                # 如果视图没有时间信息，清除mask
                gaussians._deformation.clear_static_mask()
        
        rendering_results = render(view, gaussians, pipeline, background, cam_type=cam_type)
        rendering = rendering_results["render"]
        
        if cam_type != "PanopticSports":
            gt = view.original_image[0:3, :, :].cuda()
        else:
            gt = view['image'].cuda()
        
        if "psnr" in values:
            psnr_val = psnr(rendering.unsqueeze(0), gt.unsqueeze(0))
            values["psnr"].append(psnr_val.item())

        if "ssim" in values:
            ssim_val = ssim(rendering.unsqueeze(0), gt.unsqueeze(0))
            values["ssim"].append(ssim_val.item())

        if "lpips" in values:
            if lpips_model is None:
                raise ValueError("LPIPS model is not initialized")
            # LPIPS expects inputs in [-1, 1]
            render_lp = rendering.unsqueeze(0) * 2.0 - 1.0
            gt_lp = gt.unsqueeze(0) * 2.0 - 1.0
            lpips_val = lpips_model(render_lp, gt_lp).mean()
            values["lpips"].append(lpips_val.item())

    summary = {}
    for name, vals in values.items():
        summary[name] = {
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if vals else None,
            "per_view": vals,
        }

    return summary


def parse_static_ratios(raw_ratios: str):
    """Parse comma-separated static ratios (percent)."""
    if not raw_ratios:
        return None
    ratios = []
    for part in raw_ratios.split(","):
        part = part.strip()
        if not part:
            continue
        ratios.append(float(part))
    return ratios if ratios else None


def parse_metrics(raw_metrics: str):
    if not raw_metrics:
        return ["psnr"]
    metrics = [m.strip().lower() for m in raw_metrics.split(",") if m.strip()]
    return metrics if metrics else ["psnr"]


def run_sensitivity_analysis(
    dataset: ModelParams,
    hyperparam,
    iteration: int,
    pipeline: PipelineParams,
    lambda_values: list = None,
    static_ratios: list = None,
    num_time_samples: int = 50,
    output_dir: str = None,
    metrics: list = None,
    lpips_net: str = "alex",
    num_time_segments: int = None,
):
    """
    运行静止高斯球阈值敏感性分析
    
    Args:
        dataset: 模型参数
        hyperparam: 隐藏参数
        iteration: 加载的迭代次数
        pipeline: 管道参数
        lambda_values: 要测试的lambda阈值列表
        static_ratios: 静态比例列表（百分比）
        num_time_samples: 时间采样点数量
        output_dir: 输出目录
        metrics: 指标列表
        lpips_net: LPIPS网络类型
        num_time_segments: 时间段数量（None表示使用全局模式，否则使用时间段模式）
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
        
        if metrics is None:
            metrics = ["psnr"]
        metrics = [m.lower() for m in metrics]

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
        
        # 判断是否使用时间段模式
        use_time_segments = num_time_segments is not None and num_time_segments > 1
        if use_time_segments:
            print(f"   Using time-segmented analysis with {num_time_segments} segments")
            segment_data = analyzer.compute_max_deformation_per_time_segment(num_time_segments, verbose=True)
            # 对于时间段模式，使用所有时间段的最大值作为全局统计
            all_max_defs = torch.stack(segment_data['max_deformations'])
            max_deformation = all_max_defs.max(dim=0)[0]  # 每个高斯在所有时间段的最大值
        else:
            print("   Using global time analysis")
            max_deformation = analyzer.compute_max_deformation(verbose=True)
        
        # 计算统计信息（使用全局max_deformation）
        stats = {
            'min': max_deformation.min().item(),
            'max': max_deformation.max().item(),
            'mean': max_deformation.mean().item(),
            'std': max_deformation.std().item(),
            'median': max_deformation.median().item(),
            'percentiles': {
                '10%': torch.quantile(max_deformation, 0.1).item(),
                '25%': torch.quantile(max_deformation, 0.25).item(),
                '50%': torch.quantile(max_deformation, 0.5).item(),
                '75%': torch.quantile(max_deformation, 0.75).item(),
                '90%': torch.quantile(max_deformation, 0.9).item(),
                '95%': torch.quantile(max_deformation, 0.95).item(),
                '99%': torch.quantile(max_deformation, 0.99).item(),
            }
        }
        print(f"\n   Deformation Statistics:")
        print(f"   - Min:    {stats['min']:.6f}")
        print(f"   - Max:    {stats['max']:.6f}")
        print(f"   - Mean:   {stats['mean']:.6f}")
        print(f"   - Median: {stats['median']:.6f}")
        print(f"   - Std:    {stats['std']:.6f}")
        
        # 确定lambda值范围
        target_ratios = None
        lambda_values = None
        lambda_vectors = None  # 时间段模式下的lambda向量列表
        
        if use_time_segments:
            # 时间段模式：为每个目标静态比例计算每个时间段的lambda
            if static_ratios:
                target_ratios = [r for r in static_ratios if 0 < r < 100]
                if 0 in static_ratios:
                    target_ratios = [0] + target_ratios
                if 100 in static_ratios:
                    target_ratios = target_ratios + [100]
                
                if target_ratios:
                    # 为每个时间段计算lambda向量
                    lambda_vectors = []
                    for seg_idx, max_def_seg in enumerate(segment_data['max_deformations']):
                        max_def_np = max_def_seg.cpu().numpy()
                        lambdas_for_seg = np.percentile(max_def_np, target_ratios).tolist()
                        # 过滤无效值
                        lambdas_for_seg = [v for v in lambdas_for_seg if np.isfinite(v)]
                        lambda_vectors.append(lambdas_for_seg)
                    
                    # lambda_vectors 是 [num_segments, num_target_ratios] 的形状
                    # 转置为 [num_target_ratios, num_segments]，每个目标比例对应一个lambda向量
                    lambda_vectors = list(zip(*lambda_vectors))
                    print(f"\n   Computed lambda vectors for {len(target_ratios)} target ratios")
                    print(f"   Each lambda vector has {num_time_segments} values (one per time segment)")
            else:
                # 如果没有指定static_ratios，使用建议的范围
                # 为每个时间段生成建议的lambda范围，然后使用第一个时间段的lambda作为基准
                lambda_vectors = []
                for seg_idx, max_def_seg in enumerate(segment_data['max_deformations']):
                    seg_stats = {
                        'min': max_def_seg.min().item(),
                        'max': max_def_seg.max().item(),
                        'percentiles': {
                            '10%': torch.quantile(max_def_seg, 0.1).item(),
                            '99%': torch.quantile(max_def_seg, 0.99).item(),
                        }
                    }
                    min_lambda = seg_stats['percentiles']['10%']
                    max_lambda = seg_stats['percentiles']['99%']
                    if max_lambda > min_lambda * 1.01:
                        suggested = np.geomspace(min_lambda, max_lambda, 20)
                    else:
                        suggested = np.linspace(min_lambda, max_lambda, 20)
                    max_val = seg_stats['max']
                    if np.isfinite(max_val) and max_val > 0:
                        lambdas = [0.0] + list(suggested) + [max_val * 1.5]
                    else:
                        lambdas = [0.0] + list(suggested)
                    lambdas = [v for v in lambdas if np.isfinite(v)]
                    lambdas = sorted(set(lambdas))
                    lambda_vectors.append(lambdas)
                
                # 使用第一个时间段的lambda作为基准，为每个时间段使用相同的lambda值
                # 但实际使用时会在每个时间段内分别计算mask
                if lambda_vectors:
                    base_lambdas = lambda_vectors[0]
                    # 为每个目标lambda值创建一个向量（所有时间段使用相同的lambda）
                    lambda_vectors = [[l] * num_time_segments for l in base_lambdas]
                    # 同时设置lambda_values用于显示
                    lambda_values = base_lambdas
                else:
                    lambda_vectors = None
                    lambda_values = [0.0, 1e-3]
        else:
            # 全局模式：使用原来的逻辑
            if static_ratios:
                max_def_np = max_deformation.cpu().numpy()
                target_ratios = [r for r in static_ratios if 0 < r < 100]
                if 0 in static_ratios:
                    target_ratios = [0] + target_ratios
                if 100 in static_ratios:
                    target_ratios = target_ratios + [100]
                if target_ratios:
                    lambda_values = np.percentile(max_def_np, target_ratios).tolist()
            
            if lambda_values is None:
                # 使用统计信息建议lambda范围
                min_lambda = stats['percentiles']['10%']
                max_lambda = stats['percentiles']['99%']
                if max_lambda > min_lambda * 1.01:
                    suggested = np.geomspace(min_lambda, max_lambda, 20)
                else:
                    suggested = np.linspace(min_lambda, max_lambda, 20)
                # 添加0和一些额外的点以获得更完整的曲线
                max_val = stats['max']
                if np.isfinite(max_val) and max_val > 0:
                    lambda_values = [0.0] + list(suggested) + [max_val * 1.5]
                else:
                    # 如果 max 不是有限值，只使用 suggested
                    lambda_values = [0.0] + list(suggested)
            
            # 过滤掉 NaN 和 Inf 值，然后去重并排序
            lambda_values = [v for v in lambda_values if np.isfinite(v)]
            lambda_values = sorted(set(lambda_values))
            
            # 确保至少有 2 个不同的 lambda 值
            if len(lambda_values) < 2:
                print(f"Warning: Only {len(lambda_values)} lambda value(s) generated!")
                print(f"   Deformation stats: min={stats['min']:.6e}, max={stats['max']:.6e}, std={stats['std']:.6e}")
                # 如果只有一个值或没有值，使用默认范围
                if len(lambda_values) == 0:
                    lambda_values = [0.0, stats['max'] * 1.5] if np.isfinite(stats['max']) else [0.0, 1e-3]
                elif len(lambda_values) == 1:
                    # 添加一些额外的值
                    val = lambda_values[0]
                    lambda_values = [0.0, val * 0.5, val, val * 1.5, val * 2.0] if val > 0 else [0.0, 1e-4, 1e-3]
            
            print(f"\n   Testing {len(lambda_values)} lambda values")
            print(f"   Range: [{min(lambda_values):.6f}, {max(lambda_values):.6f}]")
        
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.join(dataset.model_path, "sensitivity_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存分析结果
        analysis_path = os.path.join(output_dir, "deformation_analysis.pth")
        analyzer.save_analysis(analysis_path)
        
        # 初始化LPIPS模型（如需要）
        lpips_model = None
        if "lpips" in metrics:
            import lpips
            lpips_model = lpips.LPIPS(net=lpips_net).cuda()

        # 首先计算baseline (无优化)
        print("\n[3/4] Computing baseline metrics (no optimization)...")
        gaussians._deformation.clear_static_mask()
        baseline_metrics = compute_metrics_for_views(
            test_views, gaussians, pipeline, background, cam_type, metrics, lpips_model
        )
        if "psnr" in baseline_metrics:
            print(
                f"   Baseline PSNR: {baseline_metrics['psnr']['mean']:.4f} "
                f"± {baseline_metrics['psnr']['std']:.4f}"
            )
        if "ssim" in baseline_metrics:
            print(
                f"   Baseline SSIM: {baseline_metrics['ssim']['mean']:.4f} "
                f"± {baseline_metrics['ssim']['std']:.4f}"
            )
        if "lpips" in baseline_metrics:
            print(
                f"   Baseline LPIPS: {baseline_metrics['lpips']['mean']:.4f} "
                f"± {baseline_metrics['lpips']['std']:.4f}"
            )
        
        # 对每个lambda值进行测试
        print("\n[4/4] Running sensitivity analysis...")
        results = []
        
        # 确定要测试的lambda配置
        if use_time_segments:
            if lambda_vectors:
                # 时间段模式 + 有目标比例：遍历每个lambda向量
                test_configs = [
                    (lambda_vec, target_ratios[idx] if target_ratios and idx < len(target_ratios) else None)
                    for idx, lambda_vec in enumerate(lambda_vectors)
                ]
                print(f"\n   Testing {len(test_configs)} lambda vector configurations")
            else:
                # 时间段模式但没有目标比例：使用建议的lambda（为每个时间段使用相同的值）
                # 这种情况下，lambda_values应该已经设置了
                if lambda_values is None:
                    lambda_values = [0.0, 1e-3]
                test_configs = [(lambda_val, None) for lambda_val in lambda_values]
                print(f"\n   Testing {len(test_configs)} lambda values (same for all segments)")
        else:
            # 全局模式：遍历lambda值
            test_configs = [
                (lambda_val, target_ratios[idx] if target_ratios and idx < len(target_ratios) else None)
                for idx, lambda_val in enumerate(lambda_values)
            ]
            print(f"\n   Testing {len(test_configs)} lambda values")
        
        for idx, (lambda_config, target_ratio) in enumerate(tqdm(test_configs, desc="Testing lambda configurations")):
            if use_time_segments:
                # 时间段模式：lambda_config 是一个向量（每个时间段一个lambda值）
                if isinstance(lambda_config, (list, tuple, np.ndarray)):
                    lambda_vec = list(lambda_config)
                else:
                    # 如果是标量，为所有时间段使用相同的值
                    lambda_vec = [lambda_config] * num_time_segments
                
                # 获取每个时间段的static mask（使用对应的lambda值）
                segment_mask_data = analyzer.get_static_mask_per_time_segment(
                    lambda_vec, num_time_segments, verbose=False
                )
                # 构建时间段mask字典
                time_segment_masks = {
                    tuple(seg): mask for seg, mask in zip(
                        segment_mask_data['segments'],
                        segment_mask_data['static_masks']
                    )
                }
                # 计算平均静态比例（所有时间段的平均值）
                avg_static_ratio = np.mean([
                    mask.float().mean().item() for mask in segment_mask_data['static_masks']
                ])
                # 不设置全局mask，而是在compute_metrics_for_views中根据时间动态设置
                gaussians._deformation.clear_static_mask()
                lambda_val = lambda_vec  # 保存lambda向量
            else:
                # 全局模式：获取静止掩码
                lambda_val = lambda_config
                static_mask = analyzer.get_static_mask(lambda_val)
                static_ratio = static_mask.float().mean().item()
                # 设置静止掩码
                gaussians._deformation.set_static_mask(static_mask)
                time_segment_masks = None
                avg_static_ratio = static_ratio
            
            # 计算指标
            metrics_summary = compute_metrics_for_views(
                test_views, gaussians, pipeline, background, cam_type, metrics, lpips_model,
                time_segment_masks=time_segment_masks
            )

            if use_time_segments:
                # 时间段模式：记录每个时间段的信息
                num_total = segment_mask_data['static_masks'][0].shape[0]
                # lambda_val 是一个向量，需要转换为可序列化的格式
                if isinstance(lambda_val, (list, tuple, np.ndarray)):
                    lambda_val_serializable = [float(v) for v in lambda_val]
                else:
                    lambda_val_serializable = float(lambda_val)
                
                result = {
                    'lambda': lambda_val_serializable,  # 保存lambda向量
                    'lambda_mean': float(np.mean(lambda_val)) if isinstance(lambda_val, (list, tuple, np.ndarray)) else float(lambda_val),
                    'static_ratio': avg_static_ratio,
                    'num_static': int(avg_static_ratio * num_total),
                    'num_total': num_total,
                    'metrics': metrics_summary,
                    'time_segments': {
                        'num_segments': num_time_segments,
                        'segments': segment_mask_data['segments'],
                        'lambda_per_segment': lambda_val_serializable if isinstance(lambda_val, (list, tuple, np.ndarray)) else [float(lambda_val)] * num_time_segments,
                        'static_ratios_per_segment': [
                            mask.float().mean().item() for mask in segment_mask_data['static_masks']
                        ],
                    },
                }
            else:
                result = {
                    'lambda': lambda_val,
                    'static_ratio': static_ratio,
                    'num_static': static_mask.sum().item(),
                    'num_total': static_mask.shape[0],
                    'metrics': metrics_summary,
                }
            if 'psnr' in metrics_summary:
                result['psnr_mean'] = metrics_summary['psnr']['mean']
                result['psnr_std'] = metrics_summary['psnr']['std']
                result['psnr_drop'] = (
                    baseline_metrics['psnr']['mean'] - metrics_summary['psnr']['mean']
                )
            if 'ssim' in metrics_summary:
                result['ssim_mean'] = metrics_summary['ssim']['mean']
                result['ssim_std'] = metrics_summary['ssim']['std']
            if 'lpips' in metrics_summary:
                result['lpips_mean'] = metrics_summary['lpips']['mean']
                result['lpips_std'] = metrics_summary['lpips']['std']
            if target_ratio is not None:
                result['target_static_ratio'] = target_ratio / 100.0
            elif target_ratios and idx < len(target_ratios):
                result['target_static_ratio'] = target_ratios[idx] / 100.0
            results.append(result)
        
        # 清除静止掩码
        gaussians._deformation.clear_static_mask()
        
        # 保存结果
        results_data = {
            'metrics': metrics,
            'lpips_net': lpips_net if "lpips" in metrics else None,
            'baseline_metrics': {k: {kk: vv for kk, vv in v.items() if kk != 'per_view'} for k, v in baseline_metrics.items()},
            'deformation_stats': stats,
            'target_static_ratios': target_ratios,
            'results': [{
                k: ({kk: vv for kk, vv in v.items() if kk != 'per_view'} if k == 'metrics' else v)
                for k, v in r.items()
            } for r in results],
            'model_path': dataset.model_path,
            'iteration': iteration,
            'num_time_samples': num_time_samples,
            'num_time_segments': num_time_segments if use_time_segments else None,
        }
        
        results_path = os.path.join(output_dir, "sensitivity_results.json")
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\n   Results saved to {results_path}")
        
        # 生成图表
        plot_results(results, baseline_metrics, output_dir, metrics)
        if "psnr" in metrics:
            plot_metric_curve(
                results,
                baseline_metrics['psnr']['mean'],
                output_dir,
                "psnr",
                "PSNR (dB)",
                "psnr_lambda_curve.png",
            )

            # 打印推荐的lambda值
            print("\n" + "=" * 60)
            print("Recommended Lambda Values:")
            print("=" * 60)

            # 找到PSNR下降小于0.1dB的最大静态比例
            for threshold in [0.05, 0.1, 0.2, 0.5]:
                valid_results = [r for r in results if r.get('psnr_drop') is not None and r['psnr_drop'] < threshold]
                if valid_results:
                    best = max(valid_results, key=lambda x: x['static_ratio'])
                    # 处理lambda：如果是向量，转换为标量（使用平均值）
                    lambda_val = best['lambda']
                    if isinstance(lambda_val, (list, tuple, np.ndarray)):
                        lambda_display = np.mean(lambda_val)
                        lambda_str = f"[{', '.join([f'{v:.6f}' for v in lambda_val])}] (mean: {lambda_display:.6f})"
                    else:
                        lambda_display = float(lambda_val)
                        lambda_str = f"{lambda_display:.6f}"
                    print(f"   PSNR drop < {threshold} dB: λ = {lambda_str} "
                          f"(static: {best['static_ratio']*100:.1f}%, PSNR: {best['psnr_mean']:.2f})")

        if "ssim" in metrics:
            plot_metric_curve(
                results,
                baseline_metrics['ssim']['mean'],
                output_dir,
                "ssim",
                "SSIM",
                "ssim_lambda_curve.png",
            )
        if "lpips" in metrics:
            plot_metric_curve(
                results,
                baseline_metrics['lpips']['mean'],
                output_dir,
                "lpips",
                "LPIPS (lower is better)",
                "lpips_lambda_curve.png",
            )
        
        return results, baseline_metrics


def plot_metric_curve(results, baseline_value, output_dir, metric_name, ylabel, filename):
    """生成指标-lambda曲线图"""
    lambdas_raw = [r['lambda'] for r in results]
    static_ratios = [r['static_ratio'] * 100 for r in results]
    values = [r['metrics'][metric_name]['mean'] for r in results]
    stds = [r['metrics'][metric_name]['std'] for r in results]

    # 处理lambdas：如果是向量，转换为标量（使用平均值）
    lambdas = []
    for lam in lambdas_raw:
        if isinstance(lam, (list, tuple, np.ndarray)):
            # 如果是向量，使用平均值
            lambdas.append(float(np.mean(lam)))
        elif isinstance(lam, (int, float)):
            lambdas.append(float(lam))
        else:
            # 尝试从结果中获取lambda_mean（如果存在）
            lambdas.append(float(lam) if lam is not None else 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(lambdas, values, yerr=stds, fmt='o-', capsize=3,
                color='#2E86AB', linewidth=2, markersize=6,
                label=f'{metric_name.upper()} with optimization')
    if baseline_value is not None:
        ax.axhline(y=baseline_value, color='#E94F37', linestyle='--',
                   linewidth=2, label=f'Baseline ({baseline_value:.4f})')

    # 添加静态比例作为次坐标轴
    ax2 = ax.twinx()
    ax2.plot(lambdas, static_ratios, 's--', color='#28A745', linewidth=1.5,
             markersize=5, alpha=0.7, label='Static Ratio')
    ax2.set_ylabel('Static Gaussians (%)', fontsize=11, color='#28A745')
    ax2.tick_params(axis='y', labelcolor='#28A745')
    ax2.set_ylim(0, 100)

    ax.set_xlabel('Lambda Threshold (λ)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{metric_name.upper()} vs Lambda Threshold', fontsize=13)
    ax.set_xscale('symlog', linthresh=1e-4)
    ax.set_xlim(3e-3, 3e-1)
    ax.grid(True, alpha=0.3)

    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   {metric_name.upper()} curve saved to {plot_path}")
    plt.close(fig)


def plot_results(results, baseline_metrics, output_dir, metrics):
    """生成综合可视化图表"""
    metrics = [m.lower() for m in metrics] if metrics else ["psnr"]
    lambdas_raw = [r['lambda'] for r in results]
    static_ratios = [r['static_ratio'] * 100 for r in results]
    
    # 处理lambdas：如果是向量，转换为标量（使用平均值或lambda_mean）
    lambdas = []
    for lam in lambdas_raw:
        if isinstance(lam, (list, tuple, np.ndarray)):
            # 如果是向量，使用平均值
            lambdas.append(float(np.mean(lam)))
        elif isinstance(lam, (int, float)):
            lambdas.append(float(lam))
        else:
            # 尝试从结果中获取lambda_mean
            lambdas.append(float(lam) if lam is not None else 0.0)

    # 如果包含SSIM/LPIPS，主图改为2x2：PSNR/SSIM/LPIPS/Static Ratio
    if "ssim" in metrics or "lpips" in metrics:
        panels = []
        if "psnr" in metrics:
            panels.append(("psnr", "PSNR (dB)", baseline_metrics.get("psnr", {}).get("mean")))
        if "ssim" in metrics:
            panels.append(("ssim", "SSIM", baseline_metrics.get("ssim", {}).get("mean")))
        if "lpips" in metrics:
            panels.append(("lpips", "LPIPS (lower is better)", baseline_metrics.get("lpips", {}).get("mean")))
        panels.append(("static_ratio", "Static Gaussians (%)", None))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Static Gaussian Threshold Sensitivity Analysis', fontsize=14, fontweight='bold')
        axes_flat = axes.flatten()

        colorbar_added = False
        for idx, (name, ylabel, baseline) in enumerate(panels[:4]):
            ax = axes_flat[idx]
            if name == "static_ratio":
                # 对于static_ratio图，使用lambda标量
                lambda_for_plot = [float(np.mean(lam)) if isinstance(lam, (list, tuple, np.ndarray)) else float(lam) for lam in lambdas_raw]
                ax.plot(lambda_for_plot, static_ratios, 'o-', color='#28A745', linewidth=2, markersize=4)
                ax.fill_between(lambda_for_plot, 0, static_ratios, alpha=0.3, color='#28A745')
                ax.set_ylabel('Static Gaussians (%)', fontsize=11)
                ax.set_title('Static Ratio vs Threshold', fontsize=12)
                ax.set_ylim(0, 100)
                ax.set_xlabel('Lambda Threshold (λ)', fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.set_xscale('symlog', linthresh=1e-4)
                ax.set_xlim(3e-3, 3e-1)
            else:
                values = [r['metrics'][name]['mean'] for r in results]
                if name == "lpips":
                    deltas = [v - baseline for v in values]
                    y_label = "LPIPS Increase"
                elif name == "ssim":
                    deltas = [baseline - v for v in values]
                    y_label = "SSIM Drop"
                else:
                    deltas = [baseline - v for v in values]
                    y_label = "PSNR Drop (dB)"

                scatter = ax.scatter(static_ratios, deltas, c=lambdas, cmap='viridis',
                                     s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
                if not colorbar_added:
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Lambda (λ)', fontsize=10)
                    colorbar_added = True
                ax.axhline(y=0, color='#999999', linestyle='--', linewidth=1.0, alpha=0.7)
                ax.set_xlabel('Static Gaussians (%)', fontsize=11)
                ax.set_ylabel(y_label, fontsize=11)
                ax.set_title(f'{name.upper()} Quality-Efficiency Trade-off', fontsize=12)
                ax.grid(True, alpha=0.3)

        # 若不足4个面板，隐藏多余轴
        for j in range(len(panels), 4):
            axes_flat[j].axis('off')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "sensitivity_analysis.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   Plot saved to {plot_path}")
        plt.close(fig)
        return

    # 仅PSNR时，保留原有2x2主图
    psnrs = [r['psnr_mean'] for r in results]
    psnr_stds = [r['psnr_std'] for r in results]
    psnr_drops = [r['psnr_drop'] for r in results]
    baseline_psnr = baseline_metrics.get("psnr", {}).get("mean")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Static Gaussian Threshold Sensitivity Analysis', fontsize=14, fontweight='bold')

    # 1. PSNR vs Lambda
    ax1 = axes[0, 0]
    ax1.errorbar(lambdas, psnrs, yerr=psnr_stds, fmt='o-', capsize=3,
                 color='#2E86AB', linewidth=2, markersize=4, label='PSNR')
    if baseline_psnr is not None:
        ax1.axhline(y=baseline_psnr, color='#E94F37', linestyle='--',
                    linewidth=2, label=f'Baseline ({baseline_psnr:.2f})')
    ax1.set_xlabel('Lambda Threshold (λ)', fontsize=11)
    ax1.set_ylabel('PSNR (dB)', fontsize=11)
    ax1.set_title('Rendering Quality vs Threshold', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('symlog', linthresh=1e-4)
    ax1.set_xlim(3e-3, 3e-1)

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
    # 处理lambdas：如果是向量，转换为标量（使用平均值）
    lambda_scalars = []
    for lam in lambdas:
        if isinstance(lam, (list, tuple, np.ndarray)):
            # 如果是向量，使用平均值
            lambda_scalars.append(float(np.mean(lam)))
        else:
            lambda_scalars.append(float(lam))
    
    scatter = ax4.scatter(static_ratios, psnr_drops, c=lambda_scalars, cmap='viridis',
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
    for i, (sr, pd, lam) in enumerate(zip(static_ratios, psnr_drops, lambda_scalars)):
        if i % max(1, len(lambda_scalars) // 5) == 0:  # 只标注部分点
            ax4.annotate(f'λ={lam:.1e}', (sr, pd), fontsize=7,
                         xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()

    # 保存图表
    plot_path = os.path.join(output_dir, "sensitivity_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   Plot saved to {plot_path}")
    plt.close(fig)



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
    parser.add_argument("--lambda_min", type=float, default=0.0035,
                        help="Minimum lambda value")
    parser.add_argument("--lambda_max", type=float, default=0.10,
                        help="Maximum lambda value")
    parser.add_argument("--metrics", type=str, default="psnr",
                        help="Comma-separated metrics: psnr,ssim,lpips")
    parser.add_argument("--lpips_net", type=str, default="alex",
                        help="LPIPS network: alex or vgg")
    parser.add_argument("--static_ratios", type=str, default=None,
                        help="Comma-separated static ratios in percent, e.g. 5,10,15,...,95")
    parser.add_argument("--static_ratio_step", type=int, default=None,
                        help="Step size in percent for static ratios (5 -> 5,10,...,95)")
    parser.add_argument("--num_time_segments", type=int, default=None,
                        help="Number of time segments for time-segmented analysis (None = global mode)")
    
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
    
    # 确定lambda值范围或静态比例
    static_ratios = None
    static_ratios_arg = getattr(args, 'static_ratios', None)
    static_ratio_step = getattr(args, 'static_ratio_step', None)
    if static_ratios_arg:
        static_ratios = parse_static_ratios(static_ratios_arg)
    elif static_ratio_step:
        static_ratios = list(range(static_ratio_step, 100, static_ratio_step))

    lambda_values = None
    if static_ratios is None and args.lambda_min is not None and args.lambda_max is not None:
        lambda_mid = (args.lambda_min * args.lambda_min * args.lambda_max) ** (1/3)
        lambda_values = np.geomspace(args.lambda_min, lambda_mid, int(args.num_lambdas * 2 / 3)).tolist()
        lambda_values = lambda_values + np.geomspace(lambda_mid, args.lambda_max, int(args.num_lambdas / 3)).tolist()
        lambda_values = [0.0] + lambda_values  # 始终包含0
    
    num_time_segments = getattr(args, 'num_time_segments', None)
    
    run_sensitivity_analysis(
        model.extract(args),
        hyperparam.extract(args),
        args.iteration,
        pipeline.extract(args),
        lambda_values=lambda_values,
        static_ratios=static_ratios,
        num_time_samples=num_time_samples,
        output_dir=output_dir,
        metrics=parse_metrics(getattr(args, 'metrics', None)),
        lpips_net=getattr(args, 'lpips_net', "alex"),
        num_time_segments=num_time_segments,
    )
