from dataclasses import dataclass, field
from typing import List, Tuple
import simpy

from simulator.analyzer import Analyzer
from simulator.structures import TileTask


@dataclass
class FREConfig:
    num_cores: int = 16
    base_cycles_per_gaussian: float = 2.0
    interpolation_cycles: float = 8.0
    early_stop_ratio: float = 0.3 


class FoveatedRasterEngine:
    """多分辨率光栅化引擎，基于 tile 区域调整工作量。

    注意：为避免多核并行导致 busy 时间“按核心数累加”，这里在内部记录每个
    TileTask 在 FRE 上的忙碌区间，并在仿真结束后对所有区间做并集，得到
    FRE 模块沿时间轴的实际忙碌时长（critical path），再一次性上报给 Analyzer。
    """

    def __init__(self, env: simpy.Environment, config: FREConfig, analyzer: Analyzer):
        self.env = env
        self.config = config
        self.analyzer = analyzer
        self.resource = simpy.Resource(env, capacity=config.num_cores)
        # 记录 (start_time, end_time) 区间，单位：cycles
        self._busy_intervals: List[Tuple[float, float]] = []

    def has_free_core(self) -> bool:
        return self.resource.count < self.resource.capacity

    def _region_scale(self, region: str) -> float:
        if region == "transition":
            return 0.5
        if region == "periphery":
            return 0.25
        return 1.0

    def raster_cycles(self, task: TileTask) -> float:
        scale = self._region_scale(task.region)
        core_cycles = task.num_gaussians * self.config.early_stop_ratio * self.config.base_cycles_per_gaussian * scale
        interp = self.config.interpolation_cycles/(1.0 - scale) if scale < 1.0 else 0.0
        return core_cycles + interp

    def process(self, task: TileTask):
        return self.env.process(self._run(task))

    def _run(self, task: TileTask):
        with self.resource.request() as req:
            yield req
            cycles = self.raster_cycles(task)
            start = self.env.now
            yield self.env.timeout(cycles)
            end = self.env.now
            self._busy_intervals.append((start, end))
            # 处理完成，由 WBS 通过 _process_pair 管理完成回调

    def finalize_busy(self) -> None:
        """在仿真结束后计算 FRE 模块沿时间轴的忙碌时间并上报 Analyzer。"""
        if not self._busy_intervals:
            return
        intervals = sorted(self._busy_intervals, key=lambda x: x[0])
        merged: List[Tuple[float, float]] = []
        cur_start, cur_end = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_end:
                cur_end = max(cur_end, e)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))
        total_busy = sum(e - s for s, e in merged)
        if total_busy > 0:
            self.analyzer.record_busy("fre", total_busy)
