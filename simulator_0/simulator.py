# simulator/simulator.py
"""主模拟器：事件驱动、模块化，串联 preprocess / sort / rasterize"""

import yaml
from queue import PriorityQueue
from typing import List, Any, Optional

from .event import Event, EventType, make_event
from .structures import (
    WorkloadFrame,
    SimStats,
    build_synthetic_workload,
    parse_resolution,
)
from .memory import MemorySystem, MemoryConfig
from .preprocess import PreprocessingEngine, PreprocessConfig
from .sort import SortingEngine, SortConfig
from .rasterize import RasterizingEngine, RasterizeConfig
from .workload_loader import load_workload_from_scene


class Simulator:
    """主模拟器类：离散事件驱动，三阶段流水（可扩展为 sort-raster 重叠）。"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.current_time: float = 0.0
        self._event_id = 0
        self.event_queue: PriorityQueue = PriorityQueue()
        self.stats = SimStats()
        self.workloads: List[WorkloadFrame] = []
        self._frame_start_time: float = 0.0
        self._build_workloads()
        self._build_components()

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_workloads(self) -> None:
        sim = self.config.get("simulation", {})
        algo = self.config.get("algorithm", {})
        tile_size = algo.get("tile_size", 32)
        chunk_size = self.config.get("workload", {}).get("chunk_size", 256)
        verbose = self.config.get("output", {}).get("verbose", True)

        # 优先从算法侧加载真实 workload（dataset/scene/frame + 训练好的模型）
        real_workloads = load_workload_from_scene(
            self.config, tile_size=tile_size, chunk_size=chunk_size, verbose=verbose
        )
        if real_workloads:
            self.workloads = real_workloads
            return

        # 回退：合成 workload（平均分配）
        res_str = sim.get("resolution", "1408x1080")
        width, height = parse_resolution(res_str)
        frames_cfg = sim.get("frames", 0)
        if isinstance(frames_cfg, int):
            frame_ids = [frames_cfg]
        else:
            frame_ids = [int(x) for x in str(frames_cfg).replace(" ", "").split(",")]
        num_gaussians_per_frame = self.config.get("workload", {}).get("num_gaussians", 100_000)
        for fid in frame_ids:
            wl = build_synthetic_workload(
                width=width,
                height=height,
                tile_size=tile_size,
                num_gaussians=num_gaussians_per_frame,
                chunk_size=chunk_size,
                frame_id=fid,
            )
            self.workloads.append(wl)

    def _build_components(self) -> None:
        hw = self.config.get("hardware", {})
        algo = self.config.get("algorithm", {})
        clock = hw.get("clock_frequency", 1.0)
        self.memory = MemorySystem(MemoryConfig(
            memory_bandwidth_gbps=hw.get("memory_bandwidth", 51.2),
            cache_size_bytes=hw.get("cache_size", 1048576),
            clock_frequency_ghz=clock,
        ))
        self.preprocess_engine = PreprocessingEngine(PreprocessConfig(
            num_units=hw.get("preprocessing_units", 4),
            clock_frequency_ghz=clock,
        ))
        self.preprocess_engine.configure(
            hw.get("preprocessing_units", 4),
            clock,
        )
        self.sort_engine = SortingEngine(SortConfig(
            num_units=hw.get("sorting_units", 4),
            clock_frequency_ghz=clock,
        ))
        self.sort_engine.configure(hw.get("sorting_units", 4), clock)
        self.rasterize_engine = RasterizingEngine(RasterizeConfig(
            num_units=hw.get("rasterizing_units", 4),
            tile_size=algo.get("tile_size", 32),
            subtile_size=algo.get("subtile_size", 4),
            clock_frequency_ghz=clock,
        ))
        self.rasterize_engine.configure(
            hw.get("rasterizing_units", 4),
            algo.get("tile_size", 32),
            algo.get("subtile_size", 4),
            clock,
        )

    def add_event(self, delay: float, event: Event) -> None:
        """向事件队列插入事件，时间戳 = current_time + delay。"""
        ts = self.current_time + delay
        self._event_id += 1
        self.event_queue.put((ts, self._event_id, event))

    def run(self) -> SimStats:
        """运行仿真直到事件队列为空。"""
        while not self.event_queue.empty():
            ts, _, event = self.event_queue.get()
            self.current_time = ts
            event.handle(self)
        return self.stats

    def dispatch_event(self, event: Event) -> None:
        """根据事件类型更新状态并调度后续事件。"""
        et = event.event_type
        data = event.data or {}
        frame_id = event.frame_id if event.frame_id is not None else 0
        workloads = self.workloads
        if not workloads:
            return
        workload = workloads[frame_id] if frame_id < len(workloads) else workloads[0]

        if et == EventType.SIM_START:
            self.add_event(0, make_event(EventType.PREPROCESS_START, frame_id=0, data={"frame_id": 0}))

        elif et == EventType.PREPROCESS_START:
            fid = data.get("frame_id", 0)
            self._frame_start_time = self.current_time
            wl = workloads[fid] if fid < len(workloads) else workload
            cycles = self.preprocess_engine.total_cycles(wl)
            self.stats.preprocess_cycles += cycles
            self.add_event(cycles, make_event(EventType.PREPROCESS_DONE, frame_id=fid, data={"cycles": cycles}))

        elif et == EventType.PREPROCESS_DONE:
            fid = data.get("frame_id", 0)
            self.add_event(0, make_event(EventType.SORT_START, frame_id=fid, data={"frame_id": fid}))

        elif et == EventType.SORT_START:
            fid = data.get("frame_id", 0)
            wl = workloads[fid] if fid < len(workloads) else workload
            cycles = self.sort_engine.total_cycles(wl)
            self.stats.sort_cycles += cycles
            self.add_event(cycles, make_event(EventType.SORT_DONE, frame_id=fid, data={"cycles": cycles}))

        elif et == EventType.SORT_DONE:
            fid = data.get("frame_id", 0)
            self.add_event(0, make_event(EventType.RENDER_START, frame_id=fid, data={"frame_id": fid}))

        elif et == EventType.RENDER_START:
            fid = data.get("frame_id", 0)
            wl = workloads[fid] if fid < len(workloads) else workload
            cycles = self.rasterize_engine.total_cycles(wl)
            self.stats.rasterize_cycles += cycles
            self.add_event(cycles, make_event(EventType.FRAME_DONE, frame_id=fid, data={"cycles": cycles}))

        elif et == EventType.FRAME_DONE:
            fid = data.get("frame_id", 0)
            frame_cycles = self.current_time - getattr(self, "_frame_start_time", self.current_time)
            self.stats.frame_times_cycles.append(frame_cycles)
            next_fid = fid + 1
            if next_fid < len(workloads):
                self.add_event(0, make_event(EventType.PREPROCESS_START, frame_id=next_fid, data={"frame_id": next_fid}))
            else:
                self.stats.total_cycles = self.current_time

    def start(self) -> None:
        """注入首次事件并运行。"""
        self.add_event(0, make_event(EventType.SIM_START))
        self.run()
