from dataclasses import dataclass
import random
import simpy

from simulator.analyzer import Analyzer
from simulator.structures import TileTask


@dataclass
class UDPEConfig:
    cull_cycles: float = 2.0               # 单高斯 cull 开销
    deform_cycles: float = 4.0             # 单高斯 deform 开销
    intersection_cycles: float = 1.0       # 单高斯 intersection 开销
    fifo_depth: int = 16
    static_ratio: float = 0.4
    quasi_ratio: float = 0.4  # 其余视为 dynamic
    culling_survival_rate: float = 0.8     # cull 后存活率，用于保守估计


class UnifiedDeformPreprocessEngine:
    """
    UDPE：以高斯为单位的前端形变/剔除建模。
    - cull 与 deform 模块并行，intersection 在存活后执行。
    - 以 chunk 中高斯数量做统计近似，体现生成 tile workload 的前端延迟。
    """

    def __init__(self, env: simpy.Environment, config: UDPEConfig, analyzer: Analyzer):
        self.env = env
        self.config = config
        self.analyzer = analyzer
        self.in_queue = simpy.Store(env, capacity=config.fifo_depth)
        self.out_queue = simpy.Store(env, capacity=config.fifo_depth)
        # 按帧去重，避免同一高斯跨多 tile 被重复预处理
        self._seen_by_frame = {}
        random.seed(0)

    def classify_counts(self, task: TileTask, effective_n: int):
        """根据标签计数或比例拆分 chunk 内的高斯数量，使用去重后的有效数量。"""
        if task.label_counts:
            total_labels = sum(task.label_counts.values())
            if total_labels > 0:
                # 按比例分配到有效数量
                static_n = int(effective_n * task.label_counts.get(0, 0) / total_labels)
                quasi_n = int(effective_n * task.label_counts.get(1, 0) / total_labels)
                dynamic_n = effective_n - static_n - quasi_n
                return static_n, quasi_n, dynamic_n
        # 无标签或标签不可用时按比例估计
        n = effective_n
        static_n = int(n * self.config.static_ratio)
        quasi_n = int(n * self.config.quasi_ratio)
        dynamic_n = max(0, n - static_n - quasi_n)
        return static_n, quasi_n, dynamic_n

    def processing_cycles(self, task: TileTask) -> float:
        """
        并行近似：
        - cull 工作量：所有高斯 N_total
        - deform 工作量：quasi+dynamic
        - 两者并行：max(cull_time, deform_time)
        - intersection：对存活高斯，使用存活率估计
        """
        # 去重：同一帧中已处理的高斯不再计入
        seen = self._seen_by_frame.setdefault(task.frame_id, set())
        if task.gaussian_ids:
            ids = [gid for gid in task.gaussian_ids if gid not in seen]
            for gid in ids:
                seen.add(gid)
            effective_n = len(ids)
        else:
            effective_n = task.num_gaussians
        if effective_n <= 0:
            return 0.0

        static_n, quasi_n, dynamic_n = self.classify_counts(task, effective_n)
        total_n = static_n + quasi_n + dynamic_n
        c = self.config
        cull_time = total_n * c.cull_cycles
        deform_time = (quasi_n + dynamic_n) * c.deform_cycles
        # 存活估计
        survive_n = int(total_n * c.culling_survival_rate)
        inter_time = survive_n * c.intersection_cycles
        return max(cull_time, deform_time) + inter_time

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
