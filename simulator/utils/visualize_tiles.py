import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from simulator.structures import WorkloadFrame


REGION_TO_VAL: Dict[str, int] = {
    "fovea": 0,
    "transition": 1,
    "periphery": 2,
}

REGION_TO_COLOR = {
    0: "#2ecc71",      # green
    1: "#f1c40f",      # yellow
    2: "#e74c3c",      # red
}


def visualize_tile_regions(workload: WorkloadFrame, outfile: str) -> None:
    """
    将 tile 的区域分类可视化并保存到 outfile。
    色彩：fovea=绿，transition=黄，periphery=红。
    """
    tile_size = workload.tile_size
    width, height = workload.width, workload.height
    ntx = width // tile_size
    nty = height // tile_size
    region_map = np.full((nty, ntx), 3, dtype=np.int32)
    gaussian_map = np.zeros((nty, ntx), dtype=np.int32)

    for tile_id, tile in workload.tiles.items():
        ty = tile_id // ntx
        tx = tile_id % ntx
        region_val = REGION_TO_VAL.get(tile.region, 2)
        region_map[ty, tx] = region_val
        gaussian_map[ty, tx] = tile.num_gaussians

    cmap = colors.ListedColormap([REGION_TO_COLOR[v] for v in sorted(REGION_TO_COLOR.keys())])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(region_map, cmap=cmap, norm=norm, origin="upper")
    ax.set_title(f"Frame {workload.frame_id} tile regions (ntx={ntx}, nty={nty})")
    ax.set_xlabel("tx")
    ax.set_ylabel("ty")

    # 覆盖高斯数用文本标注
    for ty in range(nty):
        for tx in range(ntx):
            ax.text(tx, ty, str(gaussian_map[ty, tx]), ha="center", va="center", fontsize=6, color="black")

    # 自定义图例
    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor=REGION_TO_COLOR[0], label="fovea"),
        Patch(facecolor=REGION_TO_COLOR[1], label="transition"),
        Patch(facecolor=REGION_TO_COLOR[2], label="periphery"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
