from dataclasses import dataclass, field
from typing import List, Tuple
import simpy

from simulator.analyzer import Analyzer
from simulator.structures import TileTask
from simulator.memory import MemorySystem
from typing import Optional, Set, Dict, List, Tuple


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

    def __init__(self, env: simpy.Environment, config: FREConfig, analyzer: Analyzer, memory: MemorySystem):
        self.env = env
        self.config = config
        self.analyzer = analyzer
        self.memory = memory
        self.resource = simpy.Resource(env, capacity=config.num_cores)
        # 记录 (start_time, end_time) 区间，单位：cycles
        self._busy_intervals: List[Tuple[float, float]] = []
        # 用于利用率：跨所有核心的 busy 区间时长求和（不做并集）
        self._busy_core_time_sum: float = 0.0
        # 任务级服务时间样本（用于长尾分布统计）
        self._task_service_times: List[float] = []
        # 跟踪每个core上一次处理的tile的高斯集合
        self._previous_gaussians: Dict[int, Optional[Set[int]]] = {}
        # 使用 Store 来管理可用的 core_id，确保每个请求都能正确追踪到对应的 core
        self._core_id_store = simpy.Store(env, capacity=config.num_cores)
        # 初始化所有 core_id
        for core_id in range(config.num_cores):
            self._core_id_store.put(core_id)

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
        core_cycles = 4 * task.num_gaussians * self.config.early_stop_ratio * self.config.base_cycles_per_gaussian * scale
        # 插值重建为轻量级、与高斯数量无关的固定开销（仅在降采样区域启用）
        interp = self.config.interpolation_cycles * (1 - scale)
        return core_cycles + interp

    def process(self, task: TileTask):
        return self.env.process(self._run(task))

    def _run(self, task: TileTask):
        with self.resource.request() as req:
            yield req
            # 获取 resource 后，从 Store 中获取一个 core_id
            # 这样可以确保 resource 和 core_id 的分配是同步的
            core_id = yield self._core_id_store.get()
            
            try:
                stage_start = self.env.now
                # 获取当前tile的高斯集合
                current_gaussians = set(task.gaussian_ids) if task.gaussian_ids else None
                
                # 获取上一次处理的tile的高斯集合
                previous_gaussians = self._previous_gaussians.get(core_id, None)
                
                # 计算访存延迟（考虑cache命中率）
                mem_cycles = self.memory.estimate_memory_cycles_for_tile(
                    current_gaussians, 
                    previous_gaussians, 
                    task.num_gaussians
                )

                start = self.env.now

                if mem_cycles > 0:
                    self.analyzer.record_busy("memory", mem_cycles)
                    yield self.env.timeout(mem_cycles)
                
                # 更新当前core的上一次高斯集合
                if current_gaussians is not None:
                    self._previous_gaussians[core_id] = current_gaussians
                
                # 光栅化处理周期
                cycles = self.raster_cycles(task)
                yield self.env.timeout(cycles)
                end = self.env.now
                self.analyzer.record_timeline_event(
                    "fre",
                    stage_start,
                    end,
                    frame_id=task.frame_id,
                    tile_id=task.tile_id,
                    core_id=core_id,
                    num_gaussians=task.num_gaussians,
                    region=task.region,
                )
                self._busy_intervals.append((start, end))
                st = max(0.0, end - start)
                self._busy_core_time_sum += st
                self._task_service_times.append(st)
                # 处理完成，由 WBS 通过 _process_pair 管理完成回调
            finally:
                # 处理完成后，将 core_id 放回 Store，供其他请求使用
                # 注意：这里在 resource 的 with 块内释放 core_id，确保同步
                yield self._core_id_store.put(core_id)

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

    def finalize_utilization(self, total_cycles: float) -> None:
        """记录 FRE 平均利用率 = sum(core_busy_time) / (num_cores * total_cycles)。"""
        if total_cycles <= 0 or self.config.num_cores <= 0:
            return
        util = self._busy_core_time_sum / (float(self.config.num_cores) * float(total_cycles))
        self.analyzer.record_utilization("fre", util)

    def get_task_service_times(self) -> List[float]:
        return list(self._task_service_times)
