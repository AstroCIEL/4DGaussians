from dataclasses import dataclass
import random
import simpy

from simulator.analyzer import Analyzer
from simulator.structures import TileTask


@dataclass
class UDPEConfig:
    cull_cycles: float = 2.0
    deform_cycles: float = 4.0
    intersection_cycles: float = 1.0
    fifo_depth: int = 16
    static_ratio: float = 0.4
    quasi_ratio: float = 0.4  # 其余视为 dynamic


class UnifiedDeformPreprocessEngine:
    """UDPE：基于标签的动态路由，内部以 chunk 粒度建模。"""

    def __init__(self, env: simpy.Environment, config: UDPEConfig, analyzer: Analyzer):
        self.env = env
        self.config = config
        self.analyzer = analyzer
        self.in_queue = simpy.Store(env, capacity=config.fifo_depth)
        self.out_queue = simpy.Store(env, capacity=config.fifo_depth)
        random.seed(0)

    def classify_counts(self, task: TileTask):
        """根据比例拆分 chunk 内的高斯数量。"""
        n = task.num_gaussians
        static_n = int(n * self.config.static_ratio)
        quasi_n = int(n * self.config.quasi_ratio)
        dynamic_n = max(0, n - static_n - quasi_n)
        return static_n, quasi_n, dynamic_n

    def processing_cycles(self, task: TileTask) -> float:
        static_n, quasi_n, dynamic_n = self.classify_counts(task)
        c = self.config
        # 静态：cull + intersection；微动：cull + deform + intersection；巨变：deform + cull + intersection
        static_cycles = static_n * (c.cull_cycles + c.intersection_cycles)
        quasi_cycles = quasi_n * (c.cull_cycles + c.deform_cycles + c.intersection_cycles)
        dynamic_cycles = dynamic_n * (c.deform_cycles + c.cull_cycles + c.intersection_cycles)
        return static_cycles + quasi_cycles + dynamic_cycles

    def start(self):
        return self.env.process(self._run())

    def _run(self):
        while True:
            task = yield self.in_queue.get()
            if task is None:
                # 透传结束信号
                yield self.out_queue.put(None)
                break
            cycles = self.processing_cycles(task)
            self.analyzer.record_busy("udpe", cycles)
            yield self.env.timeout(cycles)
            try:
                yield self.out_queue.put(task)
            except simpy.resources.store.StoreFull:
                self.analyzer.record_fifo_block("udpe_out_full")
                yield self.out_queue.put(task)
