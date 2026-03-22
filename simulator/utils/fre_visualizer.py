import os
from typing import List, Dict, Any, Optional


def render_fre_core_trace(
    records: List[Dict[str, Any]],
    num_tiles_x: int,
    num_tiles_y: int,
    output_path: str,
    title: Optional[str] = None,
    annotate: bool = True,
) -> None:
    if not records or not output_path or num_tiles_x <= 0 or num_tiles_y <= 0:
        return

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[fre_visualizer] matplotlib unavailable: {type(e).__name__}: {e}")
        return

    # 按 core 分组
    cores = sorted({int(r["core_id"]) for r in records})
    if not cores:
        return

    # 子图布局（按网格尺寸放大，避免文字重叠）
    ncols = min(4, len(cores))
    nrows = (len(cores) + ncols - 1) // ncols
    cell_in = max(0.18, min(0.28, 8.0 / max(num_tiles_x, num_tiles_y)))
    fig_w = max(4.0 * ncols, num_tiles_x * cell_in * ncols)
    fig_h = max(4.0 * nrows, num_tiles_y * cell_in * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=150)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    axes = np.array(axes).reshape(nrows, ncols)

    # 归一化颜色范围
    hit_values = [float(r.get("cache_hit_ratio", 0.0)) for r in records]
    vmin = min(hit_values) if hit_values else 0.0
    vmax = max(hit_values) if hit_values else 1.0
    if vmin == vmax:
        vmax = vmin + 1e-6

    for idx, core_id in enumerate(cores):
        ax = axes[idx // ncols, idx % ncols]
        order_map = np.full((num_tiles_y, num_tiles_x), -1, dtype=float)
        hit_map = np.full((num_tiles_y, num_tiles_x), np.nan, dtype=float)

        core_records = [r for r in records if int(r["core_id"]) == core_id]
        for r in core_records:
            x = int(r["tile_x"])
            y = int(r["tile_y"])
            if 0 <= x < num_tiles_x and 0 <= y < num_tiles_y:
                order_map[y, x] = float(r.get("order", -1))
                hit_map[y, x] = float(r.get("cache_hit_ratio", 0.0))

        im = ax.imshow(hit_map, cmap="Blues", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(f"core {core_id}")
        ax.set_xlabel("tile_x")
        ax.set_ylabel("tile_y")
        ax.set_xticks(range(num_tiles_x))
        ax.set_yticks(range(num_tiles_y))
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)

        # 标注顺序
        if annotate:
            for y in range(num_tiles_y):
                for x in range(num_tiles_x):
                    if order_map[y, x] >= 0:
                        ax.text(x, y, f"{int(order_map[y, x])}", ha="center", va="center", fontsize=5, color="white")

    # 清理多余子图
    total_axes = nrows * ncols
    for j in range(len(cores), total_axes):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle(title or "FRE Core Tile Order & Cache Hit Rate", fontsize=12)
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_label("cache_hit_ratio")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def render_fre_core_assignment(
    records: List[Dict[str, Any]],
    num_tiles_x: int,
    num_tiles_y: int,
    output_path: str,
    title: Optional[str] = None,
) -> None:
    if not records or not output_path or num_tiles_x <= 0 or num_tiles_y <= 0:
        return

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[fre_visualizer] matplotlib unavailable: {type(e).__name__}: {e}")
        return

    core_ids = sorted({int(r["core_id"]) for r in records})
    if not core_ids:
        return

    core_index = {cid: idx for idx, cid in enumerate(core_ids)}
    assignment = np.full((num_tiles_y, num_tiles_x), -1, dtype=float)
    duplicates = 0

    for r in records:
        x = int(r["tile_x"])
        y = int(r["tile_y"])
        if 0 <= x < num_tiles_x and 0 <= y < num_tiles_y:
            idx = core_index.get(int(r["core_id"]), -1)
            if idx < 0:
                continue
            if assignment[y, x] >= 0:
                duplicates += 1
                continue
            assignment[y, x] = float(idx)

    fig_w = max(6.0, num_tiles_x * 0.25)
    fig_h = max(5.0, num_tiles_y * 0.25)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    cmap = plt.get_cmap("tab10", len(core_ids) + 1)
    assignment_for_draw = assignment.copy()
    assignment_for_draw[assignment_for_draw < 0] = len(core_ids)
    im = ax.imshow(
        assignment_for_draw,
        cmap=cmap,
        origin="upper",
        vmin=0,
        vmax=max(0, len(core_ids)),
    )
    ax.set_xlabel("tile_x")
    ax.set_ylabel("tile_y")
    ax.set_xticks(range(num_tiles_x))
    ax.set_yticks(range(num_tiles_y))
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)
    ax.set_title(title or "FRE Core Assignment (Frame 0)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("core_id")
    cbar.set_ticks(list(range(len(core_ids) + 1)))
    cbar.set_ticklabels([str(cid) for cid in core_ids] + ["unassigned"])

    unassigned = int((assignment < 0).sum())
    if duplicates > 0 or unassigned > 0:
        note = f"duplicates: {duplicates}, unassigned: {unassigned}"
        ax.text(
            0.01,
            0.01,
            note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
