from dataclasses import dataclass
import simpy
from typing import List

from simulator.analyzer import Analyzer
from simulator.structures import TileTask
from simulator.rasterize_fre import FoveatedRasterEngine
from simulator.sort_hse import HierarchicalSortEngine


@dataclass
class WBSConfig:
    window_size: int = 8
    fifo_depth: int = 32


class WorkloadBalancingScheduler:
    """
    窗口化空间局部性调度，管理 HSE 和 FRE 核的解耦调度。
    HSE 和 FRE 核独立调度，实现流水线处理：
    1. HSE 核有空闲时，分配 TileTask 进行排序
    2. 排序完成的 TileTask 进入中间队列
    3. FRE 核有空闲时，从中间队列取出 TileTask 进行光栅化
    这样可以提高硬件利用率，实现 HSE 和 FRE 之间的流水线。
    """

    def __init__(
        self,
        env: simpy.Environment,
        config: WBSConfig,
        sort_engine: HierarchicalSortEngine,
        raster_engine: FoveatedRasterEngine,
        analyzer: Analyzer,
    ):
        self.env = env
        self.config = config
        self.sort_engine = sort_engine
        self.raster_engine = raster_engine
        self.analyzer = analyzer
        self.in_queue = simpy.Store(env, capacity=config.fifo_depth)
        self.pending: List[TileTask] = []  # 等待排序的任务队列
        self.sorted_queue: List[TileTask] = []  # 排序完成等待光栅化的任务队列
        self.upstream_done = False
        self.hse_in_flight = 0  # HSE 核正在处理的任务数
        self.fre_in_flight = 0  # FRE 核正在处理的任务数

    def start(self):
        return self.env.process(self._run())

    def _dispatch_hse(self):
        """尝试把窗口内 workload 最大的任务分配给空闲的 HSE 核。"""
        while self.pending and self.sort_engine.has_free_core():
            window = self.pending[: self.config.window_size]
            idx = max(range(len(window)), key=lambda i: window[i].num_gaussians)
            task = self.pending.pop(idx)
            self.hse_in_flight += 1
            # 启动 HSE 核进行排序
            self.env.process(self._process_sort(task))

    def _dispatch_fre(self):
        """尝试把排序完成的任务分配给空闲的 FRE 核。"""
        while self.sorted_queue and self.raster_engine.has_free_core():
            task = self.sorted_queue.pop(0)  # FIFO：先排序完成的先光栅化
            self.fre_in_flight += 1
            # 启动 FRE 核进行光栅化
            self.env.process(self._process_raster(task))

    def _process_sort(self, task: TileTask):
        """HSE 核处理排序任务。"""
        yield self.sort_engine.process(task)
        # 排序完成，加入等待光栅化的队列
        self.sorted_queue.append(task)
        self.hse_in_flight -= 1
        # 尝试立即分发到 FRE 核
        self._dispatch_fre()

    def _process_raster(self, task: TileTask):
        """FRE 核处理光栅化任务。"""
        yield self.raster_engine.process(task)
        # 光栅化完成
        self.fre_in_flight -= 1

    def _run(self):
        while True:
            # 检查是否所有任务都完成
            if self.upstream_done and not self.pending and not self.sorted_queue and self.hse_in_flight == 0 and self.fre_in_flight == 0:
                break

            # 尝试分发任务到 HSE 和 FRE 核
            self._dispatch_hse()
            self._dispatch_fre()
            
            if self.upstream_done and not self.pending and not self.sorted_queue and self.hse_in_flight == 0 and self.fre_in_flight == 0:
                break

            # 等待新任务或排序完成的任务
            # 使用 any_of 同时等待输入队列和排序完成事件
            events = [self.in_queue.get()]
            ret = yield self.env.any_of(events)
            
            if events[0] in ret:
                task = ret[events[0]]
                if task is None:
                    self.upstream_done = True
                else:
                    self.pending.append(task)
            
            # 再次尝试分发
            self._dispatch_hse()
            self._dispatch_fre()
