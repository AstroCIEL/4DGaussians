# gscore_simulator/rasterizer.py (No JIT, PyTorch Vectorized Final Version)

import torch
import numpy as np
from tqdm import tqdm
from gscore_simulator.structures import RenderMetrics
import math

# All Numba-related imports are removed

def rasterize_tiles(
    sorted_gaussians_by_tile: dict,
    camera,
    config: dict,
    device # Explicitly receive GPU device
):
    """
    VRU (Volume Rendering Unit) Simulation Function (PyTorch Vectorized Version).
    Process all operations directly on GPU using PyTorch tensors to maximize performance.
    """
    
    H, W = camera.image_height, camera.image_width
    tile_size = config['tile_size']
    subtile_res = config['subtile_res']
    
    # Create buffer to store final image in GPU memory
    image_buffer = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    alpha_buffer = torch.zeros((H, W), dtype=torch.float32, device=device)
    
    metrics = RenderMetrics()
    # Performance metric calculation is omitted for now to reduce complexity
    
    print("Running Vectorized VRU (PyTorch on GPU): Rasterizing tiles...")
    
    # Tile loop runs sequentially
    for tile_id, chunks in tqdm(sorted_gaussians_by_tile.items(), desc="Rasterizing Tiles"):
        if not chunks:
            continue
            
        metrics.tiles_rendered += 1
        
        # --- Chunk and Gaussian loop (maintained for sequential alpha blending) ---
        for chunk in chunks:
            for gaus in chunk:
                # Convert gaus object data to PyTorch GPU tensor (may be NumPy)
                gaus_mean = gaus.mean.to(device) if isinstance(gaus.mean, torch.Tensor) else torch.tensor(gaus.mean, dtype=torch.float32, device=device)
                gaus_cov = gaus.cov.to(device) if isinstance(gaus.cov, torch.Tensor) else torch.tensor(gaus.cov, dtype=torch.float32, device=device)
                gaus_color = gaus.color_precomp.to(device) if isinstance(gaus.color_precomp, torch.Tensor) else torch.tensor(gaus.color_precomp, dtype=torch.float32, device=device)
                obb_corners = gaus.obb_corners.to(device) if isinstance(gaus.obb_corners, torch.Tensor) else torch.tensor(gaus.obb_corners, dtype=torch.float32, device=device)

                # 1. Define pixel area affected by Gaussian (Bounding Box)
                min_bound_orig = torch.floor(torch.min(obb_corners, dim=0).values).int()
                max_bound_orig = torch.ceil(torch.max(obb_corners, dim=0).values).int()
                
                # Clip to stay within screen boundaries
                min_bound = torch.maximum(torch.tensor([0,0], device=device), min_bound_orig)
                max_bound = torch.minimum(torch.tensor([W,H], device=device), max_bound_orig)

                if max_bound[0] <= min_bound[0] or max_bound[1] <= min_bound[1]:
                    continue
                
                # 2. Create pixel coordinate grid for the region
                px_range = torch.arange(min_bound[0], max_bound[0], device=device)
                py_range = torch.arange(min_bound[1], max_bound[1], device=device)
                # PyTorch's meshgrid uses 'ij' indexing by default, so py_grid comes first
                py_grid, px_grid = torch.meshgrid(py_range, px_range, indexing='ij')

                # 3. Vectorized alpha value calculation
                d_grid = torch.stack((px_grid, py_grid), dim=-1) - gaus_mean
                
                try:
                    cov_inv = torch.linalg.inv(gaus_cov)
                except torch.linalg.LinAlgError:
                    continue
                
                power = -0.5 * torch.einsum('hwi,ij,hwj->hw', d_grid, cov_inv, d_grid)
                alpha_map = gaus.opacity * torch.exp(power)
                
                # 4. Vectorized Subtile Skipping
                subtile_size = tile_size / subtile_res
                tile_tx = (min_bound[0] // tile_size) * tile_size
                tile_ty = (min_bound[1] // tile_size) * tile_size
                
                subtile_idx_x = ((px_grid - tile_tx) // subtile_size).long()
                subtile_idx_y = ((py_grid - tile_ty) // subtile_size).long()
                subtile_indices = (subtile_idx_y * subtile_res + subtile_idx_x)
                
                bitmap = gaus.subtile_bitmaps.get(tile_id, 0)
                
                skip_mask = ((bitmap >> subtile_indices) & 1) == 0
                alpha_map[skip_mask] = 0

                # 5. Vectorized alpha blending
                alpha_slice = alpha_buffer[min_bound[1]:max_bound[1], min_bound[0]:max_bound[0]]
                
                alpha_map[alpha_slice > 0.99] = 0

                valid_alpha_mask = alpha_map > 1e-4
                if not torch.any(valid_alpha_mask):
                    continue

                T = alpha_map * (1.0 - alpha_slice)
                T_valid = T[valid_alpha_mask]
                
                img_slice = image_buffer[min_bound[1]:max_bound[1], min_bound[0]:max_bound[0]]
                # Use unsqueeze(-1) in PyTorch to match dimensions
                img_slice[valid_alpha_mask] += T_valid.unsqueeze(-1) * gaus_color
                
                alpha_slice += T

    # Convert final result to NumPy array by moving to CPU
    rendered_image = image_buffer.cpu().numpy()
    return rendered_image, metrics