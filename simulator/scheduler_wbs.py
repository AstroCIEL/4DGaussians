from dataclasses import dataclass
import simpy
from typing import List

from simulator.analyzer import Analyzer
from simulator.structures import TileTask
from simulator.rasterize_fre import FoveatedRasterEngine


@dataclass
class WBSConfig:
    window_size: int = 8
    fifo_depth: int = 32


class WorkloadBalancingScheduler:
    """窗口化空间局部性调度，选择窗口内 workload 最大的 tile 给空闲核心。"""

    def __init__(self, env: simpy.Environment, config: WBSConfig, raster_engine: FoveatedRasterEngine, analyzer: Analyzer):
        self.env = env
        self.config = config
        self.raster_engine = raster_engine
        self.analyzer = analyzer
        self.in_queue = simpy.Store(env, capacity=config.fifo_depth)
        self.pending: List[TileTask] = []
        self.upstream_done = False
        self.in_flight = 0

    def start(self):
        return self.env.process(self._run())

    def _dispatch(self):
        """尝试把窗口内 workload 最大的任务送入光栅核心。"""
        while self.pending and self.raster_engine.has_free_core():
            window = self.pending[: self.config.window_size]
            idx = max(range(len(window)), key=lambda i: window[i].num_gaussians)
            task = self.pending.pop(idx)
            self.in_flight += 1
            self.env.process(self._wrap_raster(task))

    def _wrap_raster(self, task: TileTask):
        yield self.raster_engine.process(task)
        # 完成回调
        self.in_flight -= 1

    def _run(self):
        done_queue = self.raster_engine.done_queue
        while True:
            if self.upstream_done and not self.pending and self.in_flight == 0:
                break

            # 如果可以分发，优先分发
            self._dispatch()
            if self.upstream_done and not self.pending and self.in_flight == 0:
                break

            # 等待新任务或光栅完成
            events = [self.in_queue.get(), done_queue.get()]
            ret = yield self.env.any_of(events)
            if events[0] in ret:
                task = ret[events[0]]
                if task is None:
                    self.upstream_done = True
                else:
                    self.pending.append(task)
            if events[1] in ret:
                # 光栅完成信号已在 _wrap_raster 更新 in_flight
                pass

            self._dispatch()
