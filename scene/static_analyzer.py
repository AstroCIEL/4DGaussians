"""
Static Gaussian Analyzer - 离线分析高斯球的静止性

通过在多个时间点计算每个高斯球的形变量,
计算最大形变范数,用于判断高斯球是否静止。
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple


class StaticGaussianAnalyzer:
    """分析训练后的高斯球是否静止的分析器"""
    
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
            
            # 计算位置形变量
            dx = (means3D_deformed - means3D).norm(dim=-1)
            max_dx = torch.maximum(max_dx, dx)
            
            # 计算尺度形变量 (在log空间)
            ds = (scales_deformed - scales).norm(dim=-1)
            max_ds = torch.maximum(max_ds, ds)
            
            # 计算旋转形变量 (四元数差异)
            dr = (rotations_deformed - rotations).norm(dim=-1)
            max_dr = torch.maximum(max_dr, dr)
        
        # 综合考虑位置、尺度、旋转的形变量
        # 位置形变是最主要的指标
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
        
        # 在log空间均匀采样
        if min_lambda > 0:
            suggested_lambdas = np.geomspace(min_lambda, max_lambda, 20)
        else:
            suggested_lambdas = np.linspace(0, max_lambda, 20)
            
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
