"""
Static Gaussian Analyzer - 离线分析高斯球的静止性

通过在多个时间点计算每个高斯球的形变量,
计算最大形变范数,用于判断高斯球是否静止。
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple
from utils.general_utils import build_scaling_rotation


class StaticGaussianAnalyzer:
    """分析训练后的高斯球是否静止的分析器"""
    
    @staticmethod
    def _compute_kl_divergence(
        mean1: torch.Tensor,
        scales1: torch.Tensor,
        rotations1: torch.Tensor,
        mean2: torch.Tensor,
        scales2: torch.Tensor,
        rotations2: torch.Tensor,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """
        计算两个3D高斯分布之间的对称KL散度
        
        KL散度公式：KL(P₁||P₂) = 0.5 * [tr(Σ₂⁻¹Σ₁) + (μ₂ - μ₁)ᵀΣ₂⁻¹(μ₂ - μ₁) - k + ln(det(Σ₂)/det(Σ₁))]
        使用对称KL散度：D_KL = 0.5 * (KL(P₁||P₂) + KL(P₂||P₁))
        
        Args:
            mean1: (N, 3) 第一个高斯分布的均值
            scales1: (N, 3) 第一个高斯分布的尺度 (log空间)
            rotations1: (N, 4) 第一个高斯分布的旋转四元数
            mean2: (N, 3) 第二个高斯分布的均值
            scales2: (N, 3) 第二个高斯分布的尺度 (log空间)
            rotations2: (N, 4) 第二个高斯分布的旋转四元数
            epsilon: 数值稳定性参数
            
        Returns:
            kl_divergences: (N,) 每个高斯分布的对称KL散度
        """
        # scales 是 log 空间的，需要应用 exp 获得实际尺度值
        scales1_exp = torch.exp(scales1)  # (N, 3)
        scales2_exp = torch.exp(scales2)  # (N, 3)
        
        # 构建协方差矩阵
        # L = R @ diag(s), Σ = L @ L^T = R @ diag(s²) @ R^T
        L1 = build_scaling_rotation(scales1_exp, rotations1)  # (N, 3, 3)
        L2 = build_scaling_rotation(scales2_exp, rotations2)  # (N, 3, 3)
        
        cov1 = L1 @ L1.transpose(1, 2)  # (N, 3, 3)
        cov2 = L2 @ L2.transpose(1, 2)  # (N, 3, 3)
        
        # 添加正则化以确保数值稳定性
        eye = torch.eye(3, device=cov1.device, dtype=cov1.dtype).unsqueeze(0)  # (1, 3, 3)
        cov1_reg = cov1 + epsilon * eye
        cov2_reg = cov2 + epsilon * eye
        
        k = 3  # 维度
        
        # 计算 KL(P₁||P₂) 和 KL(P₂||P₁)
        kl_12_list = []
        kl_21_list = []
        
        for i in range(cov1.shape[0]):
            try:
                # 计算 KL(P₁||P₂)
                cov2_inv = torch.linalg.inv(cov2_reg[i])  # (3, 3)
                
                # tr(Σ₂⁻¹Σ₁)
                trace_term = torch.trace(cov2_inv @ cov1_reg[i])
                
                # (μ₂ - μ₁)ᵀΣ₂⁻¹(μ₂ - μ₁)
                mean_diff = mean2[i] - mean1[i]  # (3,)
                quad_term = mean_diff @ cov2_inv @ mean_diff
                
                # ln(det(Σ₂)/det(Σ₁))
                det1 = torch.linalg.det(cov1_reg[i])
                det2 = torch.linalg.det(cov2_reg[i])
                # 确保行列式为正
                det1 = torch.clamp(det1, min=epsilon)
                det2 = torch.clamp(det2, min=epsilon)
                log_det_term = torch.log(det2 / det1)
                
                kl_12 = 0.5 * (trace_term + quad_term - k + log_det_term)
                
                # 计算 KL(P₂||P₁)
                cov1_inv = torch.linalg.inv(cov1_reg[i])  # (3, 3)
                
                # tr(Σ₁⁻¹Σ₂)
                trace_term = torch.trace(cov1_inv @ cov2_reg[i])
                
                # (μ₁ - μ₂)ᵀΣ₁⁻¹(μ₁ - μ₂)
                mean_diff = mean1[i] - mean2[i]  # (3,)
                quad_term = mean_diff @ cov1_inv @ mean_diff
                
                # ln(det(Σ₁)/det(Σ₂))
                log_det_term = torch.log(det1 / det2)
                
                kl_21 = 0.5 * (trace_term + quad_term - k + log_det_term)
                
                kl_12_list.append(kl_12)
                kl_21_list.append(kl_21)
                
            except RuntimeError:
                # 如果矩阵求逆失败，使用简化近似
                # 使用均值差和Frobenius范数的组合
                mean_diff_norm = (mean1[i] - mean2[i]).norm() ** 2
                cov_diff_norm = (cov1_reg[i] - cov2_reg[i]).norm('fro') ** 2
                kl_approx = 0.5 * (mean_diff_norm + cov_diff_norm)
                kl_12_list.append(kl_approx)
                kl_21_list.append(kl_approx)
        
        kl_12 = torch.stack(kl_12_list)  # (N,)
        kl_21 = torch.stack(kl_21_list)  # (N,)
        
        # 计算对称KL散度
        kl_symmetric = 0.5 * (kl_12 + kl_21)
        
        # 确保非负（数值稳定性）
        kl_symmetric = torch.clamp(kl_symmetric, min=epsilon)
        
        return kl_symmetric  # (N,)
    
    def __init__(self, gaussians, num_time_samples: int = 50):
        """
        Args:
            gaussians: GaussianModel实例
            num_time_samples: 时间采样点数量
        """
        self.gaussians = gaussians
        self.num_time_samples = num_time_samples
        self._max_deformation = None  # 缓存最大形变量
        
    @torch.no_grad()
    def compute_max_deformation(self, verbose: bool = True) -> torch.Tensor:
        """
        计算每个高斯球在所有时间点的最大形变量
        
        Returns:
            max_deformation: (N,) 每个高斯球的最大形变范数
        """
        if self._max_deformation is not None:
            return self._max_deformation
            
        means3D = self.gaussians._xyz
        scales = self.gaussians._scaling
        rotations = self.gaussians._rotation
        opacity = self.gaussians._opacity
        shs = self.gaussians.get_features
        
        N = means3D.shape[0]
        device = means3D.device
        
        # 存储每个高斯球在所有时间点的形变量
        #max_kl_divergence = torch.zeros(N, device=device)
        max_dx = torch.zeros(N, device=device)
        max_ds = torch.zeros(N, device=device)
        max_dr = torch.zeros(N, device=device)
        
        # 在[0, 1]时间范围内均匀采样
        time_samples = torch.linspace(0, 1, self.num_time_samples, device=device)
        
        iterator = tqdm(time_samples, desc="Analyzing deformation") if verbose else time_samples
        
        for t in iterator:
            time = t.repeat(N, 1)
            
            # 通过deformation网络计算形变后的属性
            means3D_deformed, scales_deformed, rotations_deformed, _, _ = \
                self.gaussians._deformation(means3D, scales, rotations, opacity, shs, time)
            
            # 计算KL散度（综合考虑位置、尺度、旋转）
            # kl_divergence = self._compute_kl_divergence(
            #     means3D, scales, rotations,
            #     means3D_deformed, scales_deformed, rotations_deformed
            # )
            # max_kl_divergence = torch.maximum(max_kl_divergence, kl_divergence)
            
            # 同时保留单独的指标用于详细分析
            # 计算位置形变量
            dx = (means3D_deformed - means3D).norm(dim=-1)
            max_dx = torch.maximum(max_dx, dx)
            
            # 计算尺度形变量 (在log空间)
            ds = (scales_deformed - scales).norm(dim=-1)
            max_ds = torch.maximum(max_ds, ds)
            
            # 计算旋转形变量 (四元数差异)
            dr = (rotations_deformed - rotations).norm(dim=-1)
            max_dr = torch.maximum(max_dr, dr)
        
        # 使用x作为综合指标来衡量形变
        self._max_deformation = max_dx
        
        # 同时保存详细信息
        self._deformation_details = {
            'max_dx': max_dx,
            'max_ds': max_ds,
            'max_dr': max_dr,
        }
        
        return self._max_deformation
    
    def get_static_mask(self, lambda_threshold: float) -> torch.Tensor:
        """
        获取静止高斯球的掩码
        
        Args:
            lambda_threshold: 形变阈值,小于此值的高斯球被认为是静止的
            
        Returns:
            static_mask: (N,) bool tensor, True表示静止
        """
        max_deformation = self.compute_max_deformation(verbose=False)
        return max_deformation < lambda_threshold
    
    def get_statistics(self) -> dict:
        """获取形变量的统计信息"""
        max_deformation = self.compute_max_deformation(verbose=False)
        
        return {
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
    
    def suggest_lambda_range(self) -> Tuple[float, float, np.ndarray]:
        """
        根据形变量分布建议lambda阈值范围
        
        Returns:
            min_lambda, max_lambda, suggested_lambdas
        """
        stats = self.get_statistics()
        
        # 基于分位数建议阈值范围
        min_lambda = stats['percentiles']['10%']
        max_lambda = stats['percentiles']['99%']
        
        # 确保 min_lambda < max_lambda，避免采样时出现问题
        if max_lambda <= min_lambda:
            # 如果最大值和最小值相同或更小，使用 min 和 max 作为范围
            if stats['max'] > stats['min']:
                min_lambda = stats['min']
                max_lambda = stats['max']
            else:
                # 如果所有值都相同，使用一个小的默认范围
                min_lambda = 0.0
                max_lambda = max(stats['max'] * 1.1, 1e-4)
        
        # 在log空间均匀采样
        if min_lambda > 0 and max_lambda > min_lambda * 1.01:  # 确保有足够的范围
            suggested_lambdas = np.geomspace(min_lambda, max_lambda, 20)
        else:
            # 如果范围太小或包含0，使用线性采样
            suggested_lambdas = np.linspace(min_lambda, max_lambda, 20)
            
        return min_lambda, max_lambda, suggested_lambdas
    
    def save_analysis(self, path: str):
        """保存分析结果"""
        max_deformation = self.compute_max_deformation(verbose=False)
        torch.save({
            'max_deformation': max_deformation.cpu(),
            'deformation_details': {k: v.cpu() for k, v in self._deformation_details.items()},
            'statistics': self.get_statistics(),
            'num_time_samples': self.num_time_samples,
        }, path)
        print(f"Analysis saved to {path}")
    
    @classmethod
    def load_analysis(cls, path: str, gaussians) -> 'StaticGaussianAnalyzer':
        """加载之前的分析结果"""
        data = torch.load(path, map_location='cuda')
        analyzer = cls(gaussians, data['num_time_samples'])
        analyzer._max_deformation = data['max_deformation'].cuda()
        analyzer._deformation_details = {k: v.cuda() for k, v in data['deformation_details'].items()}
        return analyzer
