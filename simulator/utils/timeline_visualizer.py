import os
from typing import List, Dict, Tuple


def _pick_color(name: str) -> Tuple[float, float, float]:
    if name == "udpe":
        return (0.2, 0.6, 0.9)
    if name == "hse":
        return (0.2, 0.8, 0.4)
    if name == "fre":
        return (0.9, 0.4, 0.2)
    return (0.5, 0.5, 0.5)


def render_timeline_png(
    trace_events: List[Dict],
    output_path: str,
    max_tracks: int = 128,
    max_events: int = 50000,
) -> None:
    """
    将 Chrome Trace 的 X 事件渲染为 PNG 时间线图。
    - 每条轨道对应一个 (stage, tid) 组合
    - 只画前 max_tracks 条轨道，前 max_events 条事件，避免文件过大
    """
    if not trace_events or not output_path:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[timeline] matplotlib unavailable: {type(e).__name__}: {e}")
        return

    events = [e for e in trace_events if e.get("ph") == "X"]
    if not events:
        return

    if max_events and len(events) > max_events:
        events = events[:max_events]

    # 按轨道分组
    tracks = []
    track_index = {}
    for e in events:
        stage = str(e.get("cat") or e.get("name") or "unknown")
        tid = e.get("tid")
        key = (stage, tid)
        if key not in track_index:
            if max_tracks and len(tracks) >= max_tracks:
                continue
            track_index[key] = len(tracks)
            tracks.append(key)

    if not tracks:
        return

    # 准备绘图
    height = max(2, min(0.35 * len(tracks) + 1, 30))
    fig, ax = plt.subplots(figsize=(16, height), dpi=150)

    # 绘制事件
    for e in events:
        stage = str(e.get("cat") or e.get("name") or "unknown")
        tid = e.get("tid")
        key = (stage, tid)
        if key not in track_index:
            continue
        y = track_index[key]
        start = float(e.get("ts", 0.0))
        dur = float(e.get("dur", 0.0))
        if dur <= 0:
            continue
        color = _pick_color(stage)
        ax.broken_barh([(start, dur)], (y - 0.4, 0.8), facecolors=color, edgecolors="none")

    # 轴与标签
    labels = [f"{stage}:{tid}" for stage, tid in tracks]
    ax.set_yticks(list(range(len(tracks))))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("cycles")
    ax.set_ylabel("stage:core")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title("Simulator Timeline")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
