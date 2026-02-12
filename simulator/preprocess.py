# simulator/preprocess.py
"""预处理模块：4DGS 的 hexplane + MLP 变形、view 变换、frustum culling、subtile 相交测试的周期建模"""

from dataclasses import dataclass
from typing import Optional

from .structures import WorkloadFrame


@dataclass
class PreprocessConfig:
    """预处理阶段硬件/算法参数"""
    num_units: int  # 并行预处理单元数
    clock_frequency_ghz: float
    # 每高斯等效周期（hexplane + MLP + view + culling + intersection test）
    cycles_per_gaussian: float = 500.0


class PreprocessingEngine:
    """预处理引擎：按 workload 估算周期并返回完成时间（周期数）"""

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig(num_units=4, clock_frequency_ghz=1.0)

    def configure(self, num_units: int, clock_frequency_ghz: float, cycles_per_gaussian: float = 500.0):
        self.config = PreprocessConfig(
            num_units=num_units,
            clock_frequency_ghz=clock_frequency_ghz,
            cycles_per_gaussian=cycles_per_gaussian,
        )

    def total_cycles(self, workload: WorkloadFrame) -> float:
        """计算整帧预处理总周期。N 个高斯，P 个单元，流水线近似：ceil(N/P) * cycles_per_gaussian。"""
        N = workload.num_gaussians
        P = self.config.num_units
        if N <= 0:
            return 0.0
        # 简化：总工作量 N * cycles_per_gaussian，P 路并行
        total_work = N * self.config.cycles_per_gaussian
        return total_work / P
