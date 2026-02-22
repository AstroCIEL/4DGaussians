from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math


@dataclass
class GaussianAttr:
    """核心高斯属性，用于后续模块扩展（形变/着色）。"""
    idx: int
    position: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]
    opacity: float
    sh: Optional[List[float]] = None


@dataclass
class TileWorkload:
    """单个 tile 的工作负载：包含所属高斯及分块信息。"""
    tile_id: int
    gaussian_ids: List[int] = field(default_factory=list)
    chunk_sizes: List[int] = field(default_factory=list)
    region: str = "fovea"  # fovea | transition | periphery

    @property
    def num_gaussians(self) -> int:
        return sum(self.chunk_sizes) if self.chunk_sizes else len(self.gaussian_ids)

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_sizes)


@dataclass
class WorkloadFrame:
    """单帧 workload：tile 粒度的高斯任务与高斯属性。"""
    frame_id: int
    width: int
    height: int
    tile_size: int
    num_gaussians: int
    num_tiles: int
    tiles: Dict[int, TileWorkload] = field(default_factory=dict)
    gaussian_attrs: Dict[int, GaussianAttr] = field(default_factory=dict)


@dataclass
class TileTask:
    """流水线传递的 tile/chunk 任务。"""
    frame_id: int
    tile_id: int
    num_gaussians: int
    region: str
    chunk_index: int = 0
    gaussian_ids: Optional[List[int]] = None


@dataclass
class SimStats:
    """全局仿真统计信息。"""
    total_cycles: float = 0.0
    preprocess_cycles: float = 0.0
    sort_cycles: float = 0.0
    rasterize_cycles: float = 0.0
    memory_stall_cycles: float = 0.0
    frame_cycles: List[float] = field(default_factory=list)
    module_busy: Dict[str, float] = field(default_factory=dict)
    fifo_blocked: Dict[str, int] = field(default_factory=dict)

    def record_busy(self, module: str, cycles: float) -> None:
        self.module_busy[module] = self.module_busy.get(module, 0.0) + cycles

    def record_block(self, fifo: str) -> None:
        self.fifo_blocked[fifo] = self.fifo_blocked.get(fifo, 0) + 1

    def to_dict(self) -> dict:
        return {
            "total_cycles": self.total_cycles,
            "preprocess_cycles": self.preprocess_cycles,
            "sort_cycles": self.sort_cycles,
            "rasterize_cycles": self.rasterize_cycles,
            "memory_stall_cycles": self.memory_stall_cycles,
            "frame_cycles": self.frame_cycles,
            "module_busy": self.module_busy,
            "fifo_blocked": self.fifo_blocked,
        }


def parse_resolution(res_str: str) -> Tuple[int, int]:
    """解析 '1408x1080' -> (1408, 1080)。"""
    w, h = res_str.strip().lower().split("x")
    return int(w), int(h)


def build_synthetic_workload(
    width: int,
    height: int,
    tile_size: int,
    num_gaussians: int,
    chunk_size: int = 256,
    frame_id: int = 0,
) -> WorkloadFrame:
    """根据分辨率与总高斯数生成均匀分布的合成 workload。"""
    ntx = math.ceil(width / tile_size)
    nty = math.ceil(height / tile_size)
    num_tiles = ntx * nty
    avg = max(1, num_gaussians // num_tiles)
    remaining = num_gaussians
    tiles: Dict[int, TileWorkload] = {}
    for ty in range(nty):
        for tx in range(ntx):
            tile_id = ty * ntx + tx
            n = min(avg, remaining) if tile_id < num_tiles - 1 else remaining
            n = max(0, n)
            remaining -= n
            chunk_sizes = [min(chunk_size, n - i * chunk_size) for i in range((n + chunk_size - 1) // chunk_size)] if n > 0 else []
            tiles[tile_id] = TileWorkload(tile_id=tile_id, gaussian_ids=[], chunk_sizes=chunk_sizes, region=_classify_region(tx, ty, ntx, nty))
    return WorkloadFrame(
        frame_id=frame_id,
        width=width,
        height=height,
        tile_size=tile_size,
        num_gaussians=num_gaussians,
        num_tiles=num_tiles,
        tiles=tiles,
        gaussian_attrs={},
    )


def _classify_region(tx: int, ty: int, ntx: int, nty: int) -> str:
    """基于 tile 网格距离划分 fovea/transition/periphery。"""
    cx, cy = (ntx - 1) / 2.0, (nty - 1) / 2.0
    dx = (tx - cx) / max(ntx, 1)
    dy = (ty - cy) / max(nty, 1)
    dist = math.sqrt(dx * dx + dy * dy)
    if dist <= 0.15:
        return "fovea"
    if dist <= 0.35:
        return "transition"
    return "periphery"
