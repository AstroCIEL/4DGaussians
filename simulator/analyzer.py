import json
import os
from typing import Dict

from simulator.structures import SimStats


class Analyzer:
    """聚合各模块统计并输出结果。"""

    def __init__(self, stats: SimStats, output_path: str = "results/stats.json", verbose: bool = True):
        self.stats = stats
        self.output_path = output_path
        self.verbose = verbose
        self._stage_map = {
            "udpe": "preprocess_cycles",
            "hse": "sort_cycles",
            "fre": "rasterize_cycles",
        }

    def record_busy(self, module: str, cycles: float) -> None:
        self.stats.record_busy(module, cycles)
        field = self._stage_map.get(module)
        if field and hasattr(self.stats, field):
            setattr(self.stats, field, getattr(self.stats, field) + cycles)

    def record_fifo_block(self, name: str) -> None:
        self.stats.record_block(name)

    def finalize(self, total_cycles: float) -> None:
        self.stats.total_cycles = total_cycles
        self._dump()
        if self.verbose:
            self._print_summary()

    def _dump(self) -> None:
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.stats.to_dict(), f, indent=2)

    def _print_summary(self) -> None:
        print("[analyzer] === Simulation Summary ===")
        print(f"total_cycles      : {self.stats.total_cycles:.2f}")
        print(f"preprocess_cycles : {self.stats.preprocess_cycles:.2f}")
        print(f"sort_cycles       : {self.stats.sort_cycles:.2f}")
        print(f"rasterize_cycles  : {self.stats.rasterize_cycles:.2f}")
        if self.stats.frame_cycles:
            avg = sum(self.stats.frame_cycles) / len(self.stats.frame_cycles)
            print(f"frames            : {len(self.stats.frame_cycles)}, avg={avg:.2f} cycles")
        if self.stats.module_busy:
            print(f"module_busy       : {self.stats.module_busy}")
        if self.stats.fifo_blocked:
            print(f"fifo_blocked      : {self.stats.fifo_blocked}")
