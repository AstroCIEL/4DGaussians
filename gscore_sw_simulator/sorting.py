from collections import defaultdict
from tqdm import tqdm
import torch
import sys

# Import to create complete Gaussian2D objects required by rasterizer
from gscore_simulator.structures import Gaussian2D
from gscore_simulator.structures import RenderMetrics
from statistics import mean

def hierarchical_sort_and_group(
    culled_gaussians: dict,
    config: dict,
    metrics
):
    """
    GSU (Gaussian Sorting Unit) Simulation Function.
    - Combine CCU results to create complete Gaussian2D objects
    - Group created objects by intersecting tile ID
    - Sort objects within each tile by depth, then split into chunks
    """
    MAC_PER_BLENDING = 4
    # GPU device setting (can be specified in config)
    device = torch.device(config.get('device', 'cuda'))
    chunk_size = config.get('gsu_chunk_size', 256)

    # --- 0) Ensure GPU tensors in culled_gaussians ---
    for k, v in culled_gaussians.items():
        if not torch.is_tensor(v):
            culled_gaussians[k] = torch.as_tensor(v, device=device)
        else:
            culled_gaussians[k] = v.to(device)

    # --- Step 1: Reconstruct from flat (gauss_idx, tile_idx, bitmap, depth) info ---
    print("Running GSU: Reconstructing intersection info...")
    gauss_idx = culled_gaussians["gauss_idx"]    # (P,)
    tile_idx  = culled_gaussians["tile_idx"]     # (P,)
    bitmap    = culled_gaussians["bitmap"]       # (P, S)
    depth_vec = culled_gaussians["depth"]        # (P,)
    txty      = culled_gaussians["txty"]

    P = gauss_idx.shape[0]
    gaussians_info = defaultdict(lambda: {"depth": 0.0, "tiles": {}})
    tile_id_to_txty = {}    #tile_id - txty mapping dictionary
    # Traverse one pair at a time and store depth and subtile bitmap per Gaussian
    for p in tqdm(range(P), desc="Aggregating gaussian bitmaps"):
        g = int(gauss_idx[p].item())
        t = int(tile_idx[p].item())
        b = bitmap[p]      # (S,) subtile-level bitmap
        d = float(depth_vec[p].item())
        x0, y0 = txty[p].tolist()

        # Store depth once is sufficient
        gaussians_info[g]["depth"] = d
        # tiles map: key=tile_id, value=subtile bitmap
        gaussians_info[g].setdefault("tiles", {})[t] = {
        "bitmap": b,
        "txty":  (x0, y0)
        }
        # Store mapping info in tile_id_to_txty dictionary
        tile_id_to_txty[t] = (x0, y0)

    print(f"gaussian info (depth): {len(gaussians_info)}")

    # --- Steps 2-4 below remain unchanged ---
    print("Running GSU: Creating full Gaussian2D objects...")
    processed_gaussians = []
    for g_id, gaussians_info in tqdm(gaussians_info.items()):
        gaus_obj = Gaussian2D(
            source_id         = g_id,
            mean              = culled_gaussians["mean"][g_id],
            cov               = culled_gaussians["cov"][g_id],
            depth             = gaussians_info["depth"],
            opacity           = culled_gaussians["opacity"][g_id],
            color_precomp     = culled_gaussians["colors_precomp"][g_id],
            obb_corners       = culled_gaussians["obb_corners"][g_id],
            obb_axes          = culled_gaussians["obb_axes"][g_id],
            tiles             = gaussians_info["tiles"]
        )
        processed_gaussians.append(gaus_obj)

    print("Running GSU: Grouping Gaussian2D objects by tile...")
    gaussians_by_tile = defaultdict(list)

    for gaus_obj in tqdm(processed_gaussians, desc="Grouping by tile"):
        # Modified: use gaus_obj.tiles keys instead of intersecting_tiles
        for tile_id in gaus_obj.tiles.keys():
            gaussians_by_tile[tile_id].append(gaus_obj)
            tx, ty = tile_id_to_txty[tile_id]

            if tile_id == 42:   # Compare with same tile as shapeGS
                gaussians_per_tile = len(gaussians_by_tile[tile_id])
                metrics.gaussians_per_tile = gaussians_per_tile
                metrics.tile_coords = (tx, ty)
                print(f"\n[DEBUG] Tile ID {tile_id} at coordinates ({tx}, {ty}) intersects with {gaussians_per_tile} Gaussians.")

    
    tx_, ty_ = metrics.tile_coords
    print(f"[METRICS] Tile ({tx_}, {ty_}) has {metrics.gaussians_per_tile} gaussians.")

    counts_per_tile = [len(gaussians) for gaussians in gaussians_by_tile.values()]
    metrics.avg_gaussians_per_tile = mean(counts_per_tile) if counts_per_tile else 0
    metrics.max_gaussians_per_tile = max(counts_per_tile) if counts_per_tile else 0
    metrics.macs_per_tile = MAC_PER_BLENDING * metrics.gaussians_per_tile
    print(f"[METRICS] average gaussians per tile: {metrics.avg_gaussians_per_tile} gaussians.")
    print(f"[METRICS] Maximum gaussians per tile: {metrics.max_gaussians_per_tile} gaussians.")
    print("Running GSU: Performing hierarchical sorting...")
    sorted_and_chunked_gaussians = {}
    total_gaussians = 0
    total_chunks = 0

    # Hierarchical sorting per tile
    for tile_id, gaus_list in tqdm(gaussians_by_tile.items(), desc="Sorting & chunking"):
    # Sort Gaussians overlapping in tile n (precise sorting)
        gaus_list.sort()

        # Split overlapping Gaussians in tile n into chunks (approximate sorting)
        chunks = [
            gaus_list[i : i + chunk_size]
            for i in range(0, len(gaus_list), chunk_size)
        ]

        # Cumulative count
        total_gaussians += len(gaus_list)
        total_chunks += len(chunks)

        # tile_id → (tx, ty) mapping
        if tile_id in tile_id_to_txty:
            tx, ty = tile_id_to_txty[tile_id]
        else:
            print(f"WARNING: No txty found for tile_id {tile_id}. Skipping.")
            continue 
        
        if chunks:
            sorted_and_chunked_gaussians[tile_id] = {
                "txty": (tx, ty),
                "chunks": chunks
            }

    print(f"[SUMMARY] Total gaussians after sorting/chunking: {total_gaussians}")
    print(f"[SUMMARY] Total chunks: {total_chunks}")
    print(f"    sorted_and_chunked gaussians: {len(sorted_and_chunked_gaussians)}")

    return sorted_and_chunked_gaussians, tile_id_to_txty, metrics