import json
import os
import time
from datetime import datetime
from typing import Dict

from simulator.structures import SimStats


class Analyzer:
    """聚合各模块统计并输出结果。"""

    def __init__(
        self,
        stats: SimStats,
        output_path: str = "results/stats.json",
        verbose: bool = True,
        dump_enabled: bool = True,
    ):
        self.stats = stats
        self.output_path = output_path
        self.verbose = verbose
        self.dump_enabled = dump_enabled
        self._stage_map = {
            "udpe": "preprocess_cycles",
            "hse": "sort_cycles",
            "fre": "rasterize_cycles",
            "memory": "memory_stall_cycles",
        }
        self._start_time = None

    def start_simulation(self, config: dict) -> None:
        """记录模拟开始时间和配置。"""
        self._start_time = time.time()
        self.stats.start_time = datetime.now().isoformat()
        self.stats.config = config

    def record_busy(self, module: str, cycles: float) -> None:
        self.stats.record_busy(module, cycles)
        field = self._stage_map.get(module)
        if field and hasattr(self.stats, field):
            setattr(self.stats, field, getattr(self.stats, field) + cycles)

    def record_fifo_block(self, name: str) -> None:
        self.stats.record_block(name)

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
        
        if self.dump_enabled:
            self._dump()
        if self.verbose:
            self._print_summary()

    def _dump(self) -> None:
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.stats.to_dict(), f, indent=2, ensure_ascii=False)

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
        if self.stats.frame_cycles:
            avg = sum(self.stats.frame_cycles) / len(self.stats.frame_cycles)
            print(f"frames            : {len(self.stats.frame_cycles)}, avg={avg:.2f} cycles")
        if self.stats.frame_times:
            avg_time = sum(self.stats.frame_times) / len(self.stats.frame_times)
            ave_fps = 1 / avg_time
            print(f"frame_times       : avg={avg_time*1000:.4f} ms per frame")
            print(f"fps               : {ave_fps:.2f} fps")
        if self.stats.module_busy:
            print(f"module_busy       : {self.stats.module_busy}")
        if self.stats.fifo_blocked:
            print(f"fifo_blocked      : {self.stats.fifo_blocked}")
