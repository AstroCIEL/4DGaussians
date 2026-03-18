import json
import os
import time
from datetime import datetime
from typing import Dict

from simulator.structures import SimStats
from simulator.utils.stats_logger import append_stats_to_csv


class Analyzer:
    """聚合各模块统计并输出结果。"""

    def __init__(
        self,
        stats: SimStats,
        output_path: str = "results/stats.json",
        verbose: bool = True,
        dump_enabled: bool = True,
        relaxation_factor: float = 0.6,
    ):
        self.stats = stats
        self.output_path = output_path
        self.verbose = verbose
        self.dump_enabled = dump_enabled
        self.relaxation_factor = relaxation_factor
        self._stage_map = {
            "udpe": "preprocess_cycles",
            "hse": "sort_cycles",
            "fre": "rasterize_cycles",
            "memory": "memory_stall_cycles",
        }
        self._start_time = None
        self._timeline_enabled = False
        self._timeline_file = None
        self._timeline_image_file = None
        self._trace_events = []

    def start_simulation(self, config: dict) -> None:
        """记录模拟开始时间和配置。"""
        self._start_time = time.time()
        self.stats.start_time = datetime.now().isoformat()
        self.stats.config = config
        cfg_output = (config or {}).get("output", {}) if isinstance(config, dict) else {}
        self._timeline_enabled = bool(cfg_output.get("timeline_enabled", False))
        self._timeline_file = cfg_output.get(
            "timeline_file",
            os.path.join(os.path.dirname(self.output_path), "timeline_trace.json"),
        )
        self._timeline_image_file = cfg_output.get(
            "timeline_image_file",
            os.path.join(os.path.dirname(self.output_path), "timeline.png"),
        )

    def record_busy(self, module: str, cycles: float) -> None:
        self.stats.record_busy(module, cycles)
        field = self._stage_map.get(module)
        if field and hasattr(self.stats, field):
            setattr(self.stats, field, getattr(self.stats, field) + cycles)

    def record_utilization(self, module: str, utilization: float) -> None:
        self.stats.record_utilization(module, utilization)

    def record_task_time_stats(self, module: str, stats: dict) -> None:
        self.stats.record_task_time_stats(module, stats)

    def record_scheduling_stats(self, name: str, stats: dict) -> None:
        self.stats.record_scheduling_stats(name, stats)

    def record_cache_hit_ratio(self, ratio: float) -> None:
        self.stats.cache_hit_ratio = float(ratio)

    def record_fre_core_stats(self, stats: dict) -> None:
        self.stats.fre_core_stats = stats
        total_gaussians = 0
        for v in stats.values():
            if isinstance(v, dict):
                total_gaussians += int(v.get("total_gaussians", 0))
        self.stats.fre_total_gaussians = int(total_gaussians)

    def record_fifo_block(self, name: str) -> None:
        self.stats.record_block(name)

    def record_fifo_block_cycles(self, name: str, cycles: float) -> None:
        """记录由于队列 put/get 等待产生的阻塞时间（cycles）。"""
        # 同时记录次数与周期，便于对齐旧字段
        self.stats.record_block(name)
        self.stats.record_block_cycles(name, cycles)

    def record_timeline_event(
        self,
        stage: str,
        start: float,
        end: float,
        frame_id: int = None,
        tile_id: int = None,
        core_id: int = None,
        num_gaussians: int = None,
        region: str = None,
    ) -> None:
        if not self._timeline_enabled:
            return
        if end < start:
            start, end = end, start
        dur = max(0.0, float(end - start))
        args = {}
        if frame_id is not None:
            args["frame_id"] = int(frame_id)
        if tile_id is not None:
            args["tile_id"] = int(tile_id)
        if core_id is not None:
            args["core_id"] = int(core_id)
        if num_gaussians is not None:
            args["num_gaussians"] = int(num_gaussians)
        if region is not None:
            args["region"] = str(region)
        event = {
            "name": stage,
            "cat": stage,
            "ph": "X",
            "ts": float(start),
            "dur": dur,
            "pid": stage,
            "tid": int(core_id) if core_id is not None else stage,
            "args": args,
        }
        self._trace_events.append(event)

    def finalize(self, total_cycles: float, clock_period_ns: float = 1.0) -> None:
        """
        完成统计并计算每帧用时。
        
        Args:
            total_cycles: 总周期数
            clock_period_ns: 时钟周期（纳秒），用于计算实际时间
        """
        self.stats.total_cycles = total_cycles
        
        # 计算系统用时
        if self._start_time is not None:
            self.stats.elapsed_time = time.time() - self._start_time
        
        # 计算每帧用时（cycles * clock_period_ns，转换为秒）
        clock_period_s = clock_period_ns * 1e-9  # 纳秒转秒
        self.stats.frame_times = [cycles * clock_period_s for cycles in self.stats.frame_cycles]
        self.stats.fps = 1 / (sum(self.stats.frame_times) / len(self.stats.frame_times))
        self.stats.fps_r = self.stats.fps * self.relaxation_factor
        if self.dump_enabled:
            self._dump()
        if self.verbose:
            self._print_summary()

    def _dump(self) -> None:
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        data = self.stats.to_dict()
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # 追加到 CSV：路径可由 config.output.csv_file 控制，否则与 stats_file 同目录
        cfg_output = (self.stats.config or {}).get("output", {}) if isinstance(self.stats.config, dict) else {}
        csv_path = cfg_output.get(
            "csv_file",
            os.path.join(os.path.dirname(self.output_path), "stats_log.csv"),
        )
        try:
            append_stats_to_csv(data, csv_path)
        except Exception as e:
            # 不要因为记录 CSV 失败影响主流程
            print(f"[analyzer] warn: failed to append stats to csv ({type(e).__name__}: {e})")

        if self._timeline_enabled and self._trace_events and self._timeline_file:
            os.makedirs(os.path.dirname(self._timeline_file), exist_ok=True)
            with open(self._timeline_file, "w", encoding="utf-8") as f:
                json.dump({"traceEvents": self._trace_events}, f, indent=2, ensure_ascii=False)
        if self._timeline_enabled and self._trace_events and self._timeline_image_file:
            try:
                from simulator.utils.timeline_visualizer import render_timeline_png
                render_timeline_png(self._trace_events, self._timeline_image_file)
            except Exception as e:
                print(f"[analyzer] warn: failed to render timeline image ({type(e).__name__}: {e})")

    def _print_summary(self) -> None:
        print("[analyzer] === Simulation Summary ===")
        if self.stats.start_time:
            print(f"start_time        : {self.stats.start_time}")
        if self.stats.elapsed_time > 0:
            print(f"elapsed_time      : {self.stats.elapsed_time:.4f} seconds")
        print(f"total_cycles      : {self.stats.total_cycles:.2f}")
        print(f"preprocess_cycles : {self.stats.preprocess_cycles:.2f}")
        print(f"sort_cycles       : {self.stats.sort_cycles:.2f}")
        print(f"rasterize_cycles  : {self.stats.rasterize_cycles:.2f}")
        print(f"memory_stall_cycles: {self.stats.memory_stall_cycles:.2f}")
        if self.stats.frame_cycles:
            avg = sum(self.stats.frame_cycles) / len(self.stats.frame_cycles)
            print(f"frames            : {len(self.stats.frame_cycles)}, avg={avg:.2f} cycles")
        if self.stats.frame_times:
            avg_time = sum(self.stats.frame_times) / len(self.stats.frame_times)
            ave_fps = 1 / avg_time
            print(f"frame_times       : avg={avg_time*1000:.4f} ms per frame")
            print(f"fps               : {ave_fps:.2f} fps")
            print(f"fps_r             : {ave_fps*self.relaxation_factor:.2f} fps")
        if self.stats.module_busy:
            print(f"module_busy       : {self.stats.module_busy}")
        if getattr(self.stats, "module_utilization", None):
            if self.stats.module_utilization:
                print(f"module_utilization: {self.stats.module_utilization}")
        if getattr(self.stats, "task_time_stats", None):
            if self.stats.task_time_stats:
                print(f"task_time_stats   : {self.stats.task_time_stats}")
        if getattr(self.stats, "scheduling_stats", None):
            if self.stats.scheduling_stats:
                print(f"scheduling_stats  : {self.stats.scheduling_stats}")
        if self.stats.fifo_blocked:
            print(f"fifo_blocked      : {self.stats.fifo_blocked}")
        if getattr(self.stats, "fifo_blocked_cycles", None):
            if self.stats.fifo_blocked_cycles:
                print(f"fifo_blocked_cycles: {self.stats.fifo_blocked_cycles}")
        print(f"stats_file        : {self.output_path}")
