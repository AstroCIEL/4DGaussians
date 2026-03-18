import csv
import os
from typing import Mapping, Any, Optional


def _ensure_parent_dir(path: str) -> None:
    """确保 csv 所在目录存在。"""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _get_nested_value(data: Mapping[str, Any], path: str) -> Any:
    """
    通过点号分隔的路径从嵌套字典中提取值。
    例如: _get_nested_value(data, "config.hardware.udpe.skip_enabled")
    返回 None 如果路径不存在。
    """
    parts = path.split(".")
    current = data
    for part in parts:
        if not isinstance(current, Mapping):
            return None
        if part not in current:
            return None
        current = current[part]
    return current


def append_stats_to_csv(stats: Mapping[str, Any], csv_path: str) -> None:
    """
    将一次仿真的 stats 以一行追加到 csv 中。
    只记录指定的字段（按顺序）：
    start_time, total_cycles, fps, fps_r,
    config.hardware.udpe.skip_enabled, config.hardware.memory.read_latency_hiding_rate,
    config.hardware.wbs.scheduling_mode, config.algorithm.foveated_enabled,
    preprocess_cycles, sort_cycles, rasterize_cycles, module_utilization.fre
    """
    if not stats:
        return

    _ensure_parent_dir(csv_path)

    # 定义需要记录的字段（按顺序）
    field_names = [
        "start_time",
        "total_cycles",
        "fps",
        "fps_r",
        "config.simulation.dataset",
        "config.simulation.scene",
        "config.hardware.udpe.skip_enabled",
        "config.hardware.memory.read_latency_hiding_rate",
        "config.hardware.wbs.scheduling_mode",
        "config.hardware.wbs.window_size_k",
        "config.algorithm.foveated_enabled",
        "config.algorithm.early_stop_ratio",
        "preprocess_cycles",
        "sort_cycles",
        "rasterize_cycles",
        "module_utilization.fre",
    ]

    # 提取字段值
    row: dict[str, Any] = {}
    for field in field_names:
        if "." in field:
            # 嵌套路径
            value = _get_nested_value(stats, field)
        else:
            # 直接字段
            value = stats.get(field)
        row[field] = value

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

