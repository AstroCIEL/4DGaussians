from dataclasses import dataclass
from typing import List, Tuple
import math
import simpy

from simulator.analyzer import Analyzer
from simulator.structures import TileTask


@dataclass
class HSEConfig:
    num_cores: int = 16  # 排序核心数量，需与 FRE 核心数一致
    coarse_cycles_per_chunk: float = 4.0  # 粗排
    fine_cycles_per_chunk: float = 4.0  # 每 chunk 内双调排序近似
    early_stop_ratio: float = 0.3 


class HierarchicalSortEngine:
    """粗排 + 双调排序的近似建模，支持多核并行。

    为了得到 HSE 模块沿时间轴的实际忙碌时间（而不是所有核心忙碌时间简单相加），
    这里会记录每个任务在 HSE 上的忙碌区间，并在仿真结束后对所有区间做并集。
    """

    def __init__(self, env: simpy.Environment, config: HSEConfig, analyzer: Analyzer):
        self.env = env
        self.config = config
        self.analyzer = analyzer
        self.resource = simpy.Resource(env, capacity=config.num_cores)
        self._busy_intervals: List[Tuple[float, float]] = []

    def sort_cycles(self) -> float:
        '''
        like GSCore, the actual latency of sorting before rasterization begins
        is approximately sorting one chunk and precisely sorting one chunk
        '''
        c = self.config
        #n = max(1, task.num_gaussians * c.early_stop_ratio)
        coarse = c.coarse_cycles_per_chunk
        fine = c.fine_cycles_per_chunk
        return coarse + fine

    def has_free_core(self) -> bool:
        return self.resource.count < self.resource.capacity

    def process(self, task: TileTask):
        """处理一个 TileTask，返回进程。"""
        return self.env.process(self._run(task))

    def _run(self, task: TileTask):
        with self.resource.request() as req:
            yield req
            cycles = self.sort_cycles()
            start = self.env.now
            yield self.env.timeout(cycles)
            end = self.env.now
            self._busy_intervals.append((start, end))
            # HSE 处理完成后，任务继续传递给配对的 FRE 核
            # 这里不直接输出，而是通过 WBS 协调

    def finalize_busy(self) -> None:
        """在仿真结束后计算 HSE 模块沿时间轴的忙碌时间并上报 Analyzer。"""
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
            self.analyzer.record_busy("hse", total_busy)
