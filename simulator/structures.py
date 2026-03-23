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
    label: Optional[int] = None  # 0=静止,1=微动,2=巨变


@dataclass
class TileWorkload:
    """单个 tile 的工作负载：包含所属高斯信息（不做分块）。"""
    tile_id: int
    gaussian_ids: List[int] = field(default_factory=list)
    label_counts: Dict[int, int] = field(default_factory=dict)  # tile 粒度标签计数
    region: str = "fovea"  # fovea | transition | periphery

    @property
    def num_gaussians(self) -> int:
        return len(self.gaussian_ids)


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
    visible_ratio: float = 0.0  # 可见高斯球占整个场景总高斯球的比例
    label_counts: Dict[int, int] = field(default_factory=dict)  # frame级别的三种高斯球数量（0=静止,1=微动,2=巨变）
    culling_rate: Dict[int, float] = field(default_factory=dict)  # 三种高斯球分别被cull的比例


@dataclass
class TileTask:
    """流水线传递的 tile 任务（不做分块）。"""
    frame_id: int
    tile_id: int
    num_gaussians: int
    region: str
    gaussian_ids: Optional[List[int]] = None
    label_counts: Optional[Dict[int, int]] = None
    tile_x: Optional[int] = None
    tile_y: Optional[int] = None
    num_tiles_x: Optional[int] = None
    num_tiles_y: Optional[int] = None


@dataclass
class SimStats:
    """全局仿真统计信息。"""
    total_cycles: float = 0.0
    preprocess_cycles: float = 0.0
    sort_cycles: float = 0.0
    rasterize_cycles: float = 0.0
    memory_stall_cycles: float = 0.0
    cache_hit_ratio: float = 0.0
    total_tiles: int = 0
    total_gaussians: int = 0
    total_tile_gaussians: int = 0
    fre_total_gaussians: int = 0
    fre_makespan_imbalance_ratio: float = 0.0
    fre_gaussian_set_overlap_ratio: float = 0.0
    frame_cycles: List[float] = field(default_factory=list)
    module_busy: Dict[str, float] = field(default_factory=dict)
    # 平均硬件利用率（0~1），按“核心时间积分 / (核心数 * total_cycles)”计算
    module_utilization: Dict[str, float] = field(default_factory=dict)
    # 任务级统计（长尾分析用）：只存 summary，避免 JSON 过大
    task_time_stats: Dict[str, dict] = field(default_factory=dict)
    # 调度级统计（长尾分析用）
    scheduling_stats: Dict[str, dict] = field(default_factory=dict)
    # FRE core 级统计
    fre_core_stats: Dict[str, dict] = field(default_factory=dict)
    fifo_blocked: Dict[str, int] = field(default_factory=dict)
    # 记录队列 put/get 造成的阻塞等待（真实仿真时间的一部分），单位：cycles
    fifo_blocked_cycles: Dict[str, float] = field(default_factory=dict)
    start_time: str = ""  # 模拟开始时间（ISO 格式字符串）
    elapsed_time: float = 0.0  # 系统用时（秒）
    config: dict = field(default_factory=dict)  # 使用的配置内容
    frame_times: List[float] = field(default_factory=list)  # 每帧用时（秒，cycles * clock_period）
    fps: float = 0.0  # 帧率（帧/秒）
    fps_r: float = 0.0  # 放松后的帧率（帧/秒）

    def record_busy(self, module: str, cycles: float) -> None:
        self.module_busy[module] = self.module_busy.get(module, 0.0) + cycles

    def record_utilization(self, module: str, utilization: float) -> None:
        # clamp 到 [0,1]，避免浮点误差/异常
        u = float(utilization)
        if u < 0.0:
            u = 0.0
        if u > 1.0:
            u = 1.0
        self.module_utilization[module] = u

    def record_task_time_stats(self, module: str, stats: dict) -> None:
        self.task_time_stats[module] = stats

    def record_scheduling_stats(self, name: str, stats: dict) -> None:
        self.scheduling_stats[name] = stats

    def record_block(self, fifo: str) -> None:
        self.fifo_blocked[fifo] = self.fifo_blocked.get(fifo, 0) + 1

    def record_block_cycles(self, fifo: str, cycles: float) -> None:
        if cycles <= 0:
            return
        self.fifo_blocked_cycles[fifo] = self.fifo_blocked_cycles.get(fifo, 0.0) + float(cycles)

    def to_dict(self) -> dict:
        return {
            "total_cycles": self.total_cycles,
            "preprocess_cycles": self.preprocess_cycles,
            "sort_cycles": self.sort_cycles,
            "rasterize_cycles": self.rasterize_cycles,
            "memory_stall_cycles": self.memory_stall_cycles,
            "cache_hit_ratio": self.cache_hit_ratio,
            "total_tiles": self.total_tiles,
            "total_gaussians": self.total_gaussians,
            "total_tile_gaussians": self.total_tile_gaussians,
            "fre_total_gaussians": self.fre_total_gaussians,
            "fre_makespan_imbalance_ratio": self.fre_makespan_imbalance_ratio,
            "fre_gaussian_set_overlap_ratio": self.fre_gaussian_set_overlap_ratio,
            "frame_cycles": self.frame_cycles,
            "module_busy": self.module_busy,
            "module_utilization": self.module_utilization,
            "task_time_stats": self.task_time_stats,
            "scheduling_stats": self.scheduling_stats,
            "fre_core_stats": self.fre_core_stats,
            "fifo_blocked": self.fifo_blocked,
            "fifo_blocked_cycles": self.fifo_blocked_cycles,
            "start_time": self.start_time,
            "elapsed_time": self.elapsed_time,
            "ave_frame_times": self.frame_times[0],
            "fps": self.fps,
            "fps_r": self.fps_r,
            "config": self.config,
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
    frame_id: int = 0,
    fov_x: float = 90.0,
    foveated_enabled: bool = True,
) -> WorkloadFrame:
    """根据分辨率与总高斯数生成均匀分布的合成 workload。"""
    ntx = width // tile_size
    nty = height // tile_size
    num_tiles = ntx * nty
    avg = max(1, num_gaussians // max(num_tiles, 1))
    remaining = num_gaussians
    tiles: Dict[int, TileWorkload] = {}
    for ty in range(nty):
        for tx in range(ntx):
            tile_id = ty * ntx + tx
            n = min(avg, remaining) if tile_id < num_tiles - 1 else remaining
            n = max(0, n)
            remaining -= n
            tiles[tile_id] = TileWorkload(
                tile_id=tile_id,
                gaussian_ids=[],
                region=_classify_region(tx, ty, tile_size, width, height, fov_x=fov_x, foveated_enabled=foveated_enabled),
            )
    return WorkloadFrame(
        frame_id=frame_id,
        width=width,
        height=height,
        tile_size=tile_size,
        num_gaussians=num_gaussians,
        num_tiles=num_tiles,
        tiles=tiles,
        gaussian_attrs={},
        visible_ratio=1.0,  # 合成 workload 假设所有高斯都可见
        label_counts={0: 0, 1: 0, 2: 0},  # 合成 workload 无标签信息
        culling_rate={0: 0.0, 1: 0.0, 2: 0.0},  # 合成 workload 无 culling
    )


def _classify_region(
    tx: int,
    ty: int,
    tile_size: int,
    width: int,
    height: int,
    fov_x: float = 90.0,
    foveated_enabled: bool = True,
) -> str:
    """
    基于偏心角划分 fovea/transition/periphery。
    - tx/ty: tile 索引（从 0 开始）
    - tile_size: tile 边长（像素）
    - width/height: 画幅有效分辨率（像素）
    - fov_x: 水平视场角（度）
    - foveated_enabled: False 时直接视为 fovea（关闭多分辨率）
    """
    if not foveated_enabled:
        return "fovea"
    px = (tx + 0.5) * tile_size
    py = (ty + 0.5) * tile_size
    cx = width / 2.0
    cy = height / 2.0
    dist_pixel = math.hypot(px - cx, py - cy)
    focal_length = (width / 2.0) / math.tan(math.radians(fov_x / 2.0))
    eccentricity_angle = math.degrees(math.atan(dist_pixel / focal_length))
    if eccentricity_angle <= 18.0:
        return "fovea"
    if eccentricity_angle <= 30.0:
        return "transition"
    return "periphery"
