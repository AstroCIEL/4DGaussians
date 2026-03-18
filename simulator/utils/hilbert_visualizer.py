import os
from typing import List, Tuple, Optional


def _compute_grid_orders(num_tiles_x: int, num_tiles_y: int) -> List[List[int]]:
    if num_tiles_x <= 0 or num_tiles_y <= 0:
        return []

    try:
        from simulator.scheduler_wbs import hilbert_curve_order
    except Exception:
        # 兜底：若导入失败，返回按行主序
        return [[ty * num_tiles_x + tx for tx in range(num_tiles_x)] for ty in range(num_tiles_y)]

    n = max(num_tiles_x, num_tiles_y)
    orders = [[0 for _ in range(num_tiles_x)] for _ in range(num_tiles_y)]
    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            orders[ty][tx] = int(hilbert_curve_order(tx, ty, n))
    return orders


def _render_orders_png(
    orders: List[List[int]],
    output_path: str,
    annotate: bool = False,
    title: Optional[str] = None,
) -> None:
    if not orders or not output_path:
        return

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[hilbert] matplotlib unavailable: {type(e).__name__}: {e}")
        return

    data = np.array(orders, dtype=float)
    if data.size <= 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(data, cmap="viridis", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="hilbert order")

    if annotate and data.shape[0] <= 20 and data.shape[1] <= 20:
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                ax.text(x, y, f"{int(data[y, x])}", ha="center", va="center", fontsize=6, color="white")

    ax.set_xlabel("tile_x")
    ax.set_ylabel("tile_y")
    ax.set_title(title or "Hilbert Order Visualization")
    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def visualize_hilbert_for_grid(
    num_tiles_x: int,
    num_tiles_y: int,
    output_path: str,
    annotate: bool = False,
) -> None:
    orders = _compute_grid_orders(num_tiles_x, num_tiles_y)
    title = f"Hilbert Order ({num_tiles_x}x{num_tiles_y})"
    _render_orders_png(orders, output_path, annotate=annotate, title=title)


def visualize_hilbert_for_config(
    config_path: str,
    output_path: str,
    frame_index: int = 0,
    annotate: bool = False,
) -> None:
    import yaml
    from simulator.structures import build_synthetic_workload, parse_resolution
    from simulator.workload_loader import load_workload_from_scene

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    algo = cfg.get("algorithm", {})
    sim_cfg = cfg.get("simulation", {})
    tile_size = int(algo.get("tile_size", 32))

    workloads = load_workload_from_scene(cfg, config_path=config_path, tile_size=tile_size, verbose=False)
    if not workloads:
        res_str = sim_cfg.get("resolution", "1440x1024")
        width, height = parse_resolution(res_str)
        width = (width // tile_size) * tile_size
        height = (height // tile_size) * tile_size
        num_gaussians = cfg.get("workload", {}).get("num_gaussians", 100_000)
        fov_x = algo.get("fov_x", 90.0)
        foveated_enabled = algo.get("foveated_enabled", True)
        workloads = [
            build_synthetic_workload(
                width,
                height,
                tile_size,
                num_gaussians,
                frame_id=0,
                fov_x=fov_x,
                foveated_enabled=foveated_enabled,
            )
        ]

    if not workloads:
        return

    idx = max(0, min(frame_index, len(workloads) - 1))
    frame = workloads[idx]
    num_tiles_x = max(1, frame.width // frame.tile_size)
    num_tiles_y = max(1, frame.height // frame.tile_size)
    title = f"Hilbert Order (frame={frame.frame_id}, {num_tiles_x}x{num_tiles_y})"
    orders = _compute_grid_orders(num_tiles_x, num_tiles_y)
    _render_orders_png(orders, output_path, annotate=annotate, title=title)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hilbert order visualizer")
    parser.add_argument("--tiles-x", type=int, default=16, help="grid width (tile count)")
    parser.add_argument("--tiles-y", type=int, default=16, help="grid height (tile count)")
    parser.add_argument("--output", type=str, default="simulator/results/hilbert.png", help="output image path")
    parser.add_argument("--annotate", action="store_true", help="draw order index in each tile (small grids only)")
    parser.add_argument("--config", type=str, default="", help="config path to visualize first frame workload")
    parser.add_argument("--frame-index", type=int, default=0, help="frame index to visualize (default 0)")
    args = parser.parse_args()

    if args.config:
        visualize_hilbert_for_config(
            args.config,
            args.output,
            frame_index=args.frame_index,
            annotate=args.annotate,
        )
    else:
        visualize_hilbert_for_grid(
            args.tiles_x,
            args.tiles_y,
            args.output,
            annotate=args.annotate,
        )


if __name__ == "__main__":
    main()
