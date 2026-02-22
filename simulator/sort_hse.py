from dataclasses import dataclass
import math
import simpy

from simulator.analyzer import Analyzer
from simulator.structures import TileTask


@dataclass
class HSEConfig:
    coarse_cycles_per_chunk: float = 4.0  # 粗排
    fine_cycles_per_gaussian: float = 0.05  # 双调排序近似
    fifo_depth: int = 16


class HierarchicalSortEngine:
    """粗排 + 双调排序的近似建模。"""

    def __init__(self, env: simpy.Environment, config: HSEConfig, analyzer: Analyzer):
        self.env = env
        self.config = config
        self.analyzer = analyzer
        self.in_queue = simpy.Store(env, capacity=config.fifo_depth)
        self.out_queue = simpy.Store(env, capacity=config.fifo_depth)

    def sort_cycles(self, task: TileTask) -> float:
        n = max(1, task.num_gaussians)
        c = self.config
        coarse = c.coarse_cycles_per_chunk
        fine = c.fine_cycles_per_gaussian * math.log2(n)
        return coarse + fine * n

    def start(self):
        return self.env.process(self._run())

    def _run(self):
        while True:
            task = yield self.in_queue.get()
            if task is None:
                yield self.out_queue.put(None)
                break
            cycles = self.sort_cycles(task)
            self.analyzer.record_busy("hse", cycles)
            yield self.env.timeout(cycles)
            try:
                yield self.out_queue.put(task)
            except simpy.resources.store.StoreFull:
                self.analyzer.record_fifo_block("hse_out_full")
                yield self.out_queue.put(task)
