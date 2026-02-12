# simulator/rasterize.py
"""光栅化模块：按 tile/chunk/subtile 的周期建模，与 gscore/Neo 行为一致"""

from dataclasses import dataclass
from typing import Dict, Optional

from .structures import WorkloadFrame


@dataclass
class RasterizeConfig:
    """光栅化阶段硬件/算法参数"""
    num_units: int  # rasterizing units，每 unit 负责一个 tile
    tile_size: int
    subtile_size: int
    clock_frequency_ghz: float
    # 每个高斯在该 tile 内的混合等效周期（与 subtile 内像素数、early termination 等有关，取平均）
    cycles_per_gaussian_per_tile: float = 8.0


class RasterizingEngine:
    """光栅化引擎：按 chunk 粒度估算周期，tile 间并行由 num_units 体现"""

    def __init__(self, config: Optional[RasterizeConfig] = None):
        self.config = config or RasterizeConfig(
            num_units=4, tile_size=32, subtile_size=4, clock_frequency_ghz=1.0
        )

    def configure(self, num_units: int, tile_size: int, subtile_size: int,
                  clock_frequency_ghz: float, cycles_per_gaussian_per_tile: float = 8.0):
        self.config = RasterizeConfig(
            num_units=num_units,
            tile_size=tile_size,
            subtile_size=subtile_size,
            clock_frequency_ghz=clock_frequency_ghz,
            cycles_per_gaussian_per_tile=cycles_per_gaussian_per_tile,
        )

    def chunk_raster_cycles(self, num_gaussians_in_chunk: int) -> float:
        """一个 chunk 的光栅化周期（该 chunk 内高斯数 * 每高斯等效周期）。"""
        return num_gaussians_in_chunk * self.config.cycles_per_gaussian_per_tile

    def tile_raster_cycles(self, workload: WorkloadFrame, tile_id: int) -> float:
        """单 tile 所有 chunk 的光栅化总周期。"""
        if tile_id not in workload.tile_info:
            return 0.0
        _, _, chunk_sizes = workload.tile_info[tile_id]
        return sum(self.chunk_raster_cycles(c) for c in chunk_sizes)

    def total_cycles(self, workload: WorkloadFrame) -> float:
        """整帧光栅化总周期：各 tile 工作量之和 / 并行 unit 数。"""
        total_work = 0.0
        for tile_id in range(workload.num_tiles):
            total_work += self.tile_raster_cycles(workload, tile_id)
        R = self.config.num_units
        return total_work / R if R > 0 else total_work
