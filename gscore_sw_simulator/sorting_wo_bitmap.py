from collections import defaultdict
from tqdm import tqdm
import torch
import sys
from math import ceil
from statistics import mean

# Import to create complete Gaussian2D objects required by rasterizer
from gscore_simulator.structures import Gaussian2D
from gscore_simulator.structures import RenderMetrics


def hierarchical_sort_and_group(
    G_list: dict,
    config: dict,
    metrics
):
    """
    GSU (Gaussian Sorting Unit) Simulation Function (Bitmap Excluded Version).
    - Combine CCU results to create complete Gaussian2D objects
    - Group created objects by intersecting tile ID
    - Sort objects within each tile by depth, then split into chunks
    """
    
    # GPU device setting (can be specified in config)
    chunk_size = config.get('gsu_chunk_size', 256)

    print("Running GSU: Performing hierarchical sorting...")
    # Hierarchical sorting per tile
    tile_to_chunks = {}


    for tile_idx, data in tqdm(G_list.items(), total=len(G_list), desc="Chunking tiles"):
        origin = data["txty"]
        gauss_list = data["gaussians"]
        # Sort in ascending order by depth
        gauss_list_sorted = sorted(gauss_list, key=lambda e: e["depth"])

        # Total number of chunks
        n_chunks = ceil(len(gauss_list_sorted) / chunk_size)
        chunks = []
        for c in range(n_chunks):
            start = c * chunk_size
            end   = start + chunk_size
            chunk_gauss = gauss_list_sorted[start:end]
            chunks.append({
                "origin": origin,
                "chunk_id": c,
                "gaussians": chunk_gauss,
            })

        tile_to_chunks[tile_idx] = chunks

    print(f"[SUMMARY] Total tiles: {len(tile_to_chunks)}")
    return tile_to_chunks, metrics













'''
# Previous version
def hierarchical_sort_and_group(
    culled_gaussians: dict,
    G_list: dict,
    config: dict,
    metrics
):
    """
    GSU (Gaussian Sorting Unit) Simulation Function (Bitmap Excluded Version).
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

    # --- Step 1: Reconstruct from flat (gauss_idx, tile_idx, depth) info ---
    print("Running GSU: Reconstructing intersection info (Bitmap excluded)...")
    gauss_idx = culled_gaussians["gauss_idx"]  # (P,)
    tile_idx  = culled_gaussians["tile_idx"]   # (P,)
    depth_vec = culled_gaussians["depth"]      # (P,)
    txty      = culled_gaussians["txty"]       # (P, 2)

    P = gauss_idx.shape[0]
    # Dictionary to store each Gaussian's info (depth, intersecting tile ID set)
    gaussians_info = defaultdict(lambda: {"depth": 0.0, "tiles": set()})
    tile_id_to_txty = {}  # tile_id - txty mapping dictionary

    # Traverse one pair at a time and store depth and intersecting tiles per Gaussian
    print(f" gaussian idx shape: {P}")
    print("Running Vectorized Aggregation...")
    

    for p in tqdm(range(P), desc="Aggregating gaussian info"):
        g = int(gauss_idx[p].item())
        t = int(tile_idx[p].item())
        d = float(depth_vec[p].item())
        x0, y0 = txty[p].tolist()

        # Store depth once is sufficient
        gaussians_info[g]["depth"] = d
        # Add intersecting tile ID to tiles set
        gaussians_info[g]["tiles"].add(t)

        # Store mapping info in tile_id_to_txty dictionary (prevent duplicates)
        if t not in tile_id_to_txty:
            tile_id_to_txty[t] = (x0, y0)


    print(f"Total unique gaussians to process: {len(gaussians_info)}")

    # --- Step 2: Create complete Gaussian2D objects ---
    print("Running GSU: Creating full Gaussian2D objects...")
    processed_gaussians = []
    for g_id, g_info in tqdm(gaussians_info.items(), desc="Creating Gaussian2D objects"):
        gaus_obj = Gaussian2D(
            source_id        = g_id,
            mean             = culled_gaussians["mean"][g_id],
            cov              = culled_gaussians["cov"][g_id],
            depth            = g_info["depth"],
            opacity          = culled_gaussians["opacity"][g_id],
            color_precomp    = culled_gaussians["colors_precomp"][g_id],
            obb_corners      = culled_gaussians["obb_corners"][g_id],
            obb_axes         = culled_gaussians["obb_axes"][g_id],
            tiles            = g_info["tiles"]  # Now 'tiles' is a set of tile_ids
        )
        processed_gaussians.append(gaus_obj)

    # --- Step 3: Group Gaussian2D objects by tile ID ---
    print("Running GSU: Grouping Gaussian2D objects by tile...")
    gaussians_by_tile = defaultdict(list)

    for gaus_obj in tqdm(processed_gaussians, desc="Grouping by tile"):
        # Modified: gaus_obj.tiles is now a set of tile_ids, so iterate directly
        for tile_id in gaus_obj.tiles:
            gaussians_by_tile[tile_id].append(gaus_obj)

    # --- Step 4: Calculate statistics and hierarchical sorting ---
    # (Works same as before without logic changes)
    if 42 in gaussians_by_tile:
        tx, ty = tile_id_to_txty[42]
        gaussians_per_tile = len(gaussians_by_tile[42])
        metrics.gaussians_per_tile = gaussians_per_tile
        metrics.tile_coords = (tx, ty)
        print(f"\n[DEBUG] Tile ID 42 at coordinates ({tx}, {ty}) intersects with {gaussians_per_tile} Gaussians.")

    tx_, ty_ = metrics.tile_coords
    print(f"[METRICS] Tile ({tx_}, {ty_}) has {metrics.gaussians_per_tile} gaussians.")

    counts_per_tile = [len(gaussians) for gaussians in gaussians_by_tile.values()]
    metrics.avg_gaussians_per_tile = mean(counts_per_tile) if counts_per_tile else 0
    metrics.max_gaussians_per_tile = max(counts_per_tile) if counts_per_tile else 0
    metrics.macs_per_tile = MAC_PER_BLENDING * metrics.gaussians_per_tile
    print(f"[METRICS] average gaussians per tile: {metrics.avg_gaussians_per_tile:.2f} gaussians.")
    print(f"[METRICS] Maximum gaussians per tile: {metrics.max_gaussians_per_tile} gaussians.")
    
    print("Running GSU: Performing hierarchical sorting...")
    sorted_and_chunked_gaussians = {}
    total_gaussians = 0
    total_chunks = 0

    # Hierarchical sorting per tile
    for tile_id, gaus_list in tqdm(gaussians_by_tile.items(), desc="Sorting & chunking"):
        # Sort Gaussians overlapping in tile n (by depth)
        gaus_list.sort()

        # Split overlapping Gaussians in tile n into chunks
        chunks = [
            gaus_list[i : i + chunk_size]
            for i in range(0, len(gaus_list), chunk_size)
        ]

        # Cumulative count
        #total_gaussians += len(gaus_list)
        total_chunks += len(chunks)

        # tile_id -> (tx, ty) mapping
        if tile_id in tile_id_to_txty:
            tx, ty = tile_id_to_txty[tile_id]
        else:
            # This case should not occur
            print(f"WARNING: No txty found for tile_id {tile_id}. Skipping.")
            continue 
        
        if chunks:
            sorted_and_chunked_gaussians[tile_id] = {
                "txty": (tx, ty),
                "chunks": chunks
            }

    #print(f"[SUMMARY] Total gaussians after sorting/chunking: {total_gaussians}")
    print(f"[SUMMARY] Total chunks: {total_chunks}")
    print(f"[SUMMARY] Total tiles with gaussians: {len(sorted_and_chunked_gaussians)}")

    return sorted_and_chunked_gaussians, tile_id_to_txty, metrics
'''
