# simulator/sort.py
"""排序模块：二阶段排序（粗排按 pivot 分 chunk，细排每 chunk 内按深度）的周期建模"""

from dataclasses import dataclass
from typing import Dict, Optional

from .structures import WorkloadFrame


@dataclass
class SortConfig:
    """排序阶段硬件/算法参数"""
    num_units: int  # 并行排序单元数（tile 级并行）
    clock_frequency_ghz: float
    # 粗排：每高斯等效周期（与 tile 内高斯数相关，可近似常数）
    coarse_cycles_per_gaussian: float = 20.0
    # 细排：每 chunk 内每高斯比较/移动
    fine_cycles_per_gaussian: float = 15.0


class SortingEngine:
    """排序引擎：每 tile 粗排 + 各 chunk 细排，总周期按并行单元数折算"""

    def __init__(self, config: Optional[SortConfig] = None):
        self.config = config or SortConfig(num_units=4, clock_frequency_ghz=1.0)

    def configure(self, num_units: int, clock_frequency_ghz: float,
                  coarse_cycles_per_gaussian: float = 20.0, fine_cycles_per_gaussian: float = 15.0):
        self.config = SortConfig(
            num_units=num_units,
            clock_frequency_ghz=clock_frequency_ghz,
            coarse_cycles_per_gaussian=coarse_cycles_per_gaussian,
            fine_cycles_per_gaussian=fine_cycles_per_gaussian,
        )

    def tile_sort_cycles(self, workload: WorkloadFrame, tile_id: int) -> float:
        """单 tile 的排序总周期：粗排 + 各 chunk 细排。"""
        if tile_id not in workload.tile_info:
            return 0.0
        n_gauss, num_chunks, chunk_sizes = workload.tile_info[tile_id]
        if n_gauss <= 0:
            return 0.0
        coarse = n_gauss * self.config.coarse_cycles_per_gaussian
        fine = sum(c * self.config.fine_cycles_per_gaussian for c in chunk_sizes)
        return coarse + fine

    def total_cycles(self, workload: WorkloadFrame) -> float:
        """整帧排序总周期：各 tile 串行工作量之和 / 并行单元数。"""
        total_work = 0.0
        for tile_id in range(workload.num_tiles):
            total_work += self.tile_sort_cycles(workload, tile_id)
        P = self.config.num_units
        return total_work / P if P > 0 else total_work

    def tile_chunk_ready_offsets(self, workload: WorkloadFrame, tile_id: int) -> list:
        """返回该 tile 各 chunk 细排完成相对于 tile 排序开始的周期偏移（用于流水：chunk 就绪即可光栅化）。"""
        if tile_id not in workload.tile_info:
            return []
        n_gauss, num_chunks, chunk_sizes = workload.tile_info[tile_id]
        if num_chunks == 0:
            return []
        coarse = n_gauss * self.config.coarse_cycles_per_gaussian
        # 细排按 chunk 顺序完成
        offsets = []
        acc = coarse
        for c in chunk_sizes:
            acc += c * self.config.fine_cycles_per_gaussian
            offsets.append(acc)
        return offsets
