# simulator/structures.py
"""仿真用数据结构：workload/trace 与统计信息"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class WorkloadFrame:
    """单帧 workload：预处理后高斯数、每 tile 的高斯与 chunk 信息（用于排序/光栅化建模）"""
    frame_id: int
    num_gaussians: int  # 经过 frustum culling 后的高斯数量
    num_tiles: int
    # tile_id -> (num_gaussians_in_tile, num_chunks, gaussians_per_chunk_list)
    tile_info: Dict[int, Tuple[int, int, List[int]]] = field(default_factory=dict)


@dataclass
class SimStats:
    """单次仿真统计"""
    total_cycles: float = 0.0
    preprocess_cycles: float = 0.0
    sort_cycles: float = 0.0
    rasterize_cycles: float = 0.0
    memory_stall_cycles: float = 0.0
    frame_times_cycles: List[float] = field(default_factory=list)
    # 可选细节
    per_tile_sort_cycles: Dict[int, float] = field(default_factory=dict)
    per_tile_rasterize_cycles: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_cycles": self.total_cycles,
            "preprocess_cycles": self.preprocess_cycles,
            "sort_cycles": self.sort_cycles,
            "rasterize_cycles": self.rasterize_cycles,
            "memory_stall_cycles": self.memory_stall_cycles,
            "frame_times_cycles": self.frame_times_cycles,
            "per_tile_sort_cycles": self.per_tile_sort_cycles,
            "per_tile_rasterize_cycles": self.per_tile_rasterize_cycles,
        }


def build_synthetic_workload(
    width: int,
    height: int,
    tile_size: int,
    num_gaussians: int,
    chunk_size: int = 256,
    frame_id: int = 0,
) -> WorkloadFrame:
    """根据分辨率与总高斯数生成合成 workload（均匀分配到各 tile）。"""
    ntx = (width + tile_size - 1) // tile_size
    nty = (height + tile_size - 1) // tile_size
    num_tiles = ntx * nty
    avg_per_tile = max(1, num_gaussians // num_tiles)
    tile_info = {}
    remaining = num_gaussians
    for t in range(num_tiles):
        n = min(avg_per_tile, remaining) if t < num_tiles - 1 else remaining
        n = max(0, n)
        remaining -= n
        if n <= 0:
            chunk_list = []
            num_chunks = 0
        else:
            chunk_list = [min(chunk_size, n - i * chunk_size) for i in range((n + chunk_size - 1) // chunk_size)]
            num_chunks = len(chunk_list)
        tile_info[t] = (n, num_chunks, chunk_list)
    return WorkloadFrame(frame_id=frame_id, num_gaussians=num_gaussians, num_tiles=num_tiles, tile_info=tile_info)


def parse_resolution(res_str: str) -> Tuple[int, int]:
    """解析 '1408x1080' -> (1408, 1080)"""
    w, h = res_str.strip().lower().split("x")
    return int(w), int(h)
