import numpy as np
import torch
from tqdm import tqdm
import math
from gscore_simulator.structures import Gaussian2D
from gscore_simulator.structures import RenderMetrics


def rasterize_tiles(
    sorted_gaussians_by_tile: dict,
    tile_id_to_txty:       dict,
    camera,
    config: dict,
    device,
    metrics
):
    H, W = camera.image_height, camera.image_width
    tile_size = config['tile_size']
    subtile_res = config['subtile_res']
    subtile_size = tile_size // subtile_res
    termination_threshold = 1e-4
    last_valid_mask = None
    image_buffer = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    
    # ★ Total Gaussian counter for simulation to skip computations
    total_saved_gaussians = 0
    target = (32, 16)

    print("Running GPU-accelerated VRU: Processing each tile...")
    flag = 1
    for tile_idx, tiles_data in tqdm(sorted_gaussians_by_tile.items(), desc="Rasterizing Tiles"):
        # Chunk list already sorted by depth from sorting.py
        chunks = tiles_data["chunks"]
        txty = tiles_data["txty"]
        tx, ty = txty
        
        if not chunks:
            print(f"[DEBUG] Tile {tile_idx} at ({tx},{ty}) has no chunks. Skipping.")
            continue

        # ★ 1. Tile level setup (pixel coordinates, buffer, subtile map, etc.)
        image_h, image_w = image_buffer.shape[0], image_buffer.shape[1]
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(tile_size, device=device, dtype=torch.float32),
            torch.arange(tile_size, device=device, dtype=torch.float32),
            indexing='ij'
        )
        px_local_tile = grid_x.flatten()
        py_local_tile = grid_y.flatten()
        px_global_tile = tx  + px_local_tile
        py_global_tile = ty  + py_local_tile
        
        #print(f"[DEBUG] global tile x: {px_global_tile}, global tile y: {py_global_tile}")
        valid_pixel_mask = (px_global_tile < image_w) & (py_global_tile < image_h)
        num_valid_pixels = valid_pixel_mask.sum()
        if num_valid_pixels == 0:
            print("there's no valid pixels")
            continue
            
        # ★ 2. Buffer to track pixel states for the entire tile
        pixel_color = torch.zeros(num_valid_pixels, 3, device=device)
        pixel_T = torch.ones(num_valid_pixels, device=device) # Transmittance buffer

        subtile_x_map = torch.div(px_local_tile[valid_pixel_mask], subtile_size, rounding_mode='trunc')
        subtile_y_map = torch.div(py_local_tile[valid_pixel_mask], subtile_size, rounding_mode='trunc')
        pixel_to_subtile_map = subtile_y_map * subtile_res + subtile_x_map

        # [METRICS] Initialize cumulative sum
        gaussians_ter_tile = 0
        count = 0
        # ★ 3. Loop over depth-sorted chunks
        for chunk_idx, chunk in enumerate(chunks):
            
            # ★ 4. Early termination check for the whole tile before chunk processing
            if pixel_T.max() < termination_threshold:   #1e-4
                # [METRICS] Calculate saved_gaussians after early termination
                for k, v in tile_id_to_txty.items():
                    if v == target:
                        print(f"coordinates: {v}")
                        saved_count = 0
                        for remaining_chunk in chunks[chunk_idx:]:
                            saved_count += len(remaining_chunk)
                        
                        print("[WARNING] EARLY TERMINATION !!")
                        metrics.saved_gaussians = saved_count
                        metrics.total_gaussians = sum(len(c) for c in chunks)

                        # ✅ saved rate (%)
                        if metrics.total_gaussians > 0:
                            metrics.saved_rate_percent = round(
                                100.0 * metrics.saved_gaussians / metrics.total_gaussians, 2
                            )
                        else:
                            metrics.saved_rate_percent = 0.0

                        print(f"[METRICS] saved gaussians at {k}: {metrics.saved_gaussians}")
                        print(f"[METRICS] saved rate at {k}: {metrics.saved_rate_percent}%")
                break

            # Create tensors from current chunk data
            try:
                gaus_means = torch.stack([g.mean for g in chunk], dim=0).to(device)
                gaus_opacities = torch.tensor([g.opacity for g in chunk], device=device)
                gaus_colors = torch.stack([g.color_precomp for g in chunk], dim=0).to(device)
                gaus_bitmaps = torch.stack([g.tiles[tile_idx]['bitmap'] for g in chunk], dim=0).to(device)
                gaus_covs_cpu = torch.stack([g.cov for g in chunk], dim=0).to('cpu')
                gaus_inv_covs_cpu = torch.linalg.inv(gaus_covs_cpu)
                gaus_inv_covs = gaus_inv_covs_cpu.to(device)
            except RuntimeError:
                print(f"DEBUG: Skipping chunk in tile {tile_idx} due to tensor/linalg error.")
                continue

            # ★ 5. Blend Gaussians in current chunk in order
            for g_idx in range(len(chunk)):
                # subtile bitmap operation
                active_subtiles = (gaus_bitmaps[g_idx] == 1).nonzero().squeeze(-1)
                if active_subtiles.numel() == 0: continue
                # Test which subtile each pixel belongs to within tile x subtile bitmap operation result
                pixel_to_subtile_map = pixel_to_subtile_map.to(active_subtiles.dtype)
                pixel_mask_for_gaus = torch.isin(pixel_to_subtile_map, active_subtiles)
                pixel_mask_for_gaus &= (pixel_T > termination_threshold)
                

                if flag:
                    if g_idx == len(chunk) - 1:
                        print(f"\n⛏️ DEBUG INFO FOR LAST GAUSSIAN IN CHUNK")
                        print(f"   - mean: {mean_g}")
                        print(f"   - color: {color_g}")
                        print(f"   - opacity: {opacity_g}")
                        print(f"   - inv_cov:\n{inv_cov_g}")
                        print(f"   - active_subtiles: {active_subtiles}")
                        print(f"   - pixel_mask_for_gaus sum: {pixel_mask_for_gaus.sum()}")
                    flag = 0



                if not pixel_mask_for_gaus.any(): 
                    #print(f"pixel - subtile map : {pixel_to_subtile_map}, active subtiles: {active_subtiles}")
                    count += 1
                    continue
                
                # Perform alpha blending only for selected pixels
                px_active = px_global_tile[valid_pixel_mask][pixel_mask_for_gaus]
                py_active = py_global_tile[valid_pixel_mask][pixel_mask_for_gaus]
                
                mean_g, opacity_g, color_g, inv_cov_g = gaus_means[g_idx], gaus_opacities[g_idx], gaus_colors[g_idx], gaus_inv_covs[g_idx]
                
                dx, dy = px_active - mean_g[0], py_active - mean_g[1]
                cx, cxy, cy = inv_cov_g[0, 0], inv_cov_g[0, 1], inv_cov_g[1, 1]
                power = -0.5 * (cx * dx**2 + cy * dy**2 + 2 * cxy * dx * dy)
                alpha = opacity_g * torch.exp(torch.clamp(power, min=-15.0, max=0.0))
                #print(f"   - alpha max: {alpha.max()}, min: {alpha.min()}, shape: {alpha.shape}")

                if (alpha.max()< 1.0/255.0):
                    continue
                T_old = pixel_T[pixel_mask_for_gaus]
                pixel_color[pixel_mask_for_gaus] += (T_old * alpha).unsqueeze(-1) * color_g
                pixel_T[pixel_mask_for_gaus] *= (1.0 - alpha)
                last_valid_mask = pixel_mask_for_gaus.clone()

        #[METRICS] gaussians per tile
        #if tile_idx == 100:
        #    metrics.tile_coords = (tx, ty)
        #    gaussians_per_tile += len(chunk)

        # ... (same logic as before for writing to final result image buffer)
        print(f" pixel mask miss : {count}")
        final_r = torch.zeros_like(px_local_tile, dtype=torch.float32)
        final_g = torch.zeros_like(px_local_tile, dtype=torch.float32)
        final_b = torch.zeros_like(px_local_tile, dtype=torch.float32)
        final_r[valid_pixel_mask] = pixel_color[:, 0]
        final_g[valid_pixel_mask] = pixel_color[:, 1]
        final_b[valid_pixel_mask] = pixel_color[:, 2]
        start_y, start_x = int(ty) * int(tile_size), int(tx) * int(tile_size)
        ts = int(tile_size)
        end_y = min(start_y + ts, image_h)
        end_x = min(start_x + ts, image_w)
        h_slice = end_y - start_y
        w_slice = end_x - start_x
        #print(f" [DEBUG] start x: {start_x}, start y: {start_y}, ts: {ts}")

        image_buffer[start_y:end_y, start_x:end_x, 0] = final_r.reshape(ts, ts)[:h_slice, :w_slice]
        image_buffer[start_y:end_y, start_x:end_x, 1] = final_g.reshape(ts, ts)[:h_slice, :w_slice]
        image_buffer[start_y:end_y, start_x:end_x, 2] = final_b.reshape(ts, ts)[:h_slice, :w_slice]
        #image_buffer[start_y:start_y+ts, start_x:start_x+ts, 0] = final_r.reshape(ts, ts)
        #image_buffer[start_y:start_y+ts, start_x:start_x+ts, 1] = final_g.reshape(ts, ts)
        #image_buffer[start_y:start_y+ts, start_x:start_x+ts, 2] = final_b.reshape(ts, ts)

    tx_, ty_ = metrics.tile_coords
    #print(f"[METRICS] Tile ({tx_}, {ty_}) has {metrics.gaussians_per_tile} gaussians.")   
    print(f"[DEBUG] image_buffer at 1st tile : {image_buffer[0:15, 0:15, 0]} ")
    print(f"\nSimulation Complete. Total saved Gaussians by hierarchical sorting & early termination: {total_saved_gaussians}")
    rendered_image = image_buffer.cpu().numpy()

    return rendered_image, metrics