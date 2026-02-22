# simulator/analyzer.py
"""性能分析、报告与可视化"""

import json
import os
from typing import Optional

from .structures import SimStats


class Analyzer:
    """统计结果分析、打印与持久化。"""

    def __init__(self, stats: SimStats, config: Optional[dict] = None):
        self.stats = stats
        self.config = config or {}
        self.verbose = self.config.get("output", {}).get("verbose", True)

    def report_console(self) -> None:
        """在控制台打印汇总报告。"""
        s = self.stats
        if self.verbose:
            print("========== 4DGS Simulator 仿真报告 ==========")
            print(f"  总周期数:        {s.total_cycles:.0f} cycles")
            print(f"  预处理周期:     {s.preprocess_cycles:.0f} cycles")
            print(f"  排序周期:        {s.sort_cycles:.0f} cycles")
            print(f"  光栅化周期:      {s.rasterize_cycles:.0f} cycles")
            if s.memory_stall_cycles > 0:
                print(f"  内存 stall 周期: {s.memory_stall_cycles:.0f} cycles")
            if s.frame_times_cycles:
                print(f"  帧数:            {len(s.frame_times_cycles)}")
                print(f"  首帧周期:        {s.frame_times_cycles[0]:.0f} cycles")
            print("==============================================")

    def save_stats(self, path: Optional[str] = None) -> str:
        """将 stats 写入 JSON 文件；路径未指定时从 config.output.stats_file 读取。"""
        out_cfg = self.config.get("output", {})
        filepath = path or out_cfg.get("stats_file", "results/stats.json")
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.stats.to_dict(), f, indent=2, ensure_ascii=False)
        return filepath

    def run(self, save_path: Optional[str] = None) -> None:
        """执行报告并可选保存。"""
        self.report_console()
        if save_path is not None or self.config.get("output", {}).get("stats_file"):
            p = self.save_stats(save_path)
            if self.verbose:
                print(f"统计已写入: {p}")
