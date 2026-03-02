from dataclasses import dataclass
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
    """粗排 + 双调排序的近似建模，支持多核并行。"""

    def __init__(self, env: simpy.Environment, config: HSEConfig, analyzer: Analyzer):
        self.env = env
        self.config = config
        self.analyzer = analyzer
        self.resource = simpy.Resource(env, capacity=config.num_cores)

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
            self.analyzer.record_busy("hse", cycles)
            yield self.env.timeout(cycles)
            # HSE 处理完成后，任务继续传递给配对的 FRE 核
            # 这里不直接输出，而是通过 WBS 协调
