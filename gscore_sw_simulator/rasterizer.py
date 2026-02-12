# gscore_simulator/rasterizer.py (GPU accelerated final version)

import numpy as np
import torch
from tqdm import tqdm
from gscore_simulator.structures import RenderMetrics
import numba
from numba import cuda
import math
from gscore_simulator.structures import Gaussian2D

# ======================================================================
# === CUDA kernel functions to be executed on GPU ===
# ======================================================================

@cuda.jit
def rasterize_tile_kernel(
    # Input/output buffers (GPU memory)
    image_buffer,         # (H, W, 3) Full image buffer
    alpha_buffer,         # (H, W) Full alpha buffer
    
    # Current tile information
    tx, ty, tile_size,
    
    # Data of Gaussians belonging to this tile (GPU memory)
    gaus_means,           # (G, 2)
    gaus_inv_covs,        # (G, 2, 2)
    gaus_opacities,       # (G,)
    gaus_colors,          # (G, 3)
    gaus_bitmaps,         # (G,)
    subtile_res
):
    """
    CUDA kernel where GPU threads process pixels of a single tile in parallel.
    Each thread is responsible for one pixel.
    """
    # 1. Calculate local and global image coordinates for the pixel assigned to the current thread
    px_local, py_local = cuda.grid(2)

    tile_h, tile_w = image_buffer.shape[0], image_buffer.shape[1]
    
    px_global = tx * tile_size + px_local
    py_global = ty * tile_size + py_local

    # If thread is outside tile area, do nothing and exit immediately
    if px_local >= tile_size or py_local >= tile_size or px_global >= tile_w or py_global >= tile_h:
        return

    # 2. Initialize variables for Volume Rendering
    T = 1.0  # Transmittance
    pixel_color_r, pixel_color_g, pixel_color_b = 0.0, 0.0, 0.0
    num_gaussians = len(gaus_means)
    subtile_size = tile_size // subtile_res

    # 3. Process all Gaussians in order (front-to-back) for this pixel
    for g in range(num_gaussians):
        # 3.1. Subtile Skipping
        stx = px_local // subtile_size
        sty = py_local // subtile_size
        subtile_idx = int(sty * subtile_res + stx)
        if not ((gaus_bitmaps[g] >> subtile_idx) & 1):
            continue

        # 3.2. Calculate alpha value (Point-in-Ellipse)
        d_x = float(px_global) - gaus_means[g, 0]
        d_y = float(py_global) - gaus_means[g, 1]
        
        inv_cov = gaus_inv_covs[g]
        power = -0.5 * (inv_cov[0, 0]*d_x*d_x + inv_cov[1, 1]*d_y*d_y + 2*inv_cov[0, 1]*d_x*d_y)
        
        if power < -15.0 or power > 0:
            continue
        
        # cuda.exp is the GPU exponential function provided by numba.cuda
        alpha = gaus_opacities[g] * cuda.exp(power)
        
        # 3.3. Alpha blending
        color_contribution = T * alpha
        pixel_color_r += color_contribution * gaus_colors[g, 0]
        pixel_color_g += color_contribution * gaus_colors[g, 1]
        pixel_color_b += color_contribution * gaus_colors[g, 2]

        T = T * (1.0 - alpha)

        # 3.4. Early termination
        if T < 1e-4:
            break
            
    # 4. Write the final calculated color and alpha values to global buffers (GPU memory)
    image_buffer[py_global, px_global, 0] = pixel_color_r
    image_buffer[py_global, px_global, 1] = pixel_color_g
    image_buffer[py_global, px_global, 2] = pixel_color_b
    alpha_buffer[py_global, px_global] = 1.0 - T


def rasterize_tiles(
    sorted_gaussians_by_tile: dict,  # keys are integer tile IDs
    camera,
    config: dict,
    device  # Explicitly receive GPU device
):
    """
    VRU simulation function (GPU accelerated final version).
    Execute CUDA kernel for each tile to render the complete image.
    Keys of sorted_gaussians_by_tile are (tx, ty) pixel start coordinate tuples,
    values are lists of Gaussian2D object chunks.
    """
    H, W       = camera.image_height, camera.image_width
    tile_size  = config['tile_size']
    subtile_res= config['subtile_res']

    # Create result buffers on GPU
    image_buffer = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    alpha_buffer = torch.zeros((H, W),    dtype=torch.float32, device=device)

    metrics = RenderMetrics()

    print("Running GPU-accelerated VRU: Launching CUDA kernels for each tile...")
    #print(f" sorted gaussians: {sorted_gaussians_by_tile}")
    for tile_idx, tiles_data in tqdm(sorted_gaussians_by_tile.items(), desc="Rasterizing Tiles"):
        chunks = tiles_data["chunks"]
        txty = tiles_data["txty"]
        tx, ty = txty
        if not chunks:
            continue

        # 1) Extract only Gaussian2D objects from each chunk
        gaussians_in_tile = [gaus for chunk in chunks for gaus in chunk if isinstance(gaus, Gaussian2D)]
        #print(f"    Finishing get gaussians in chunk! {len(gaussians_in_tile)} gaussians")

        # Safety measure: skip tiles with no Gaussians to process
        if not gaussians_in_tile:
            print(f"DEBUG: Tile {tile_idx} has no gaussians to rasterize. Skipping.")
            continue

        print(f"DEBUG: Processing Tile {tile_idx} with {len(gaussians_in_tile)} gaussians.")

        mean_list = []
        opacity_list = []
        color_list = []
        cov_list = []
        bitmap_list = []

        # 2. Traverse the `gaussians_in_tile` list only once.
        for g in gaussians_in_tile:
            # Add each Gaussian's attribute to its corresponding list
            mean_list.append(g.mean)
            opacity_list.append(g.opacity)
            color_list.append(g.color_precomp)
            cov_list.append(g.cov)
            bitmap_list.append(g.tiles[tile_idx]["bitmap"])

        # 3. After the loop, convert collected lists to GPU tensors all at once.
        # This method is much more efficient than looping multiple times.

        gaus_means     = torch.stack(mean_list, dim=0).to(device)
        gaus_opacities = torch.tensor(opacity_list, device=device)
        gaus_colors    = torch.stack(color_list, dim=0).to(device)
        gaus_covs      = torch.stack(cov_list, dim=0).to(device)
        gaus_bitmaps   = torch.stack(bitmap_list, dim=0).to(device)

        #print(f"DEBUG:gaus_covs shape: {gaus_covs.shape}")
        # --- Full debugging output code as requested ---
        print("-------------------------------------------")
        print(f"DEBUG: Processing Tile {tile_idx} with {len(gaussians_in_tile)} gaussians.")
        print(f"DEBUG: gaus_means     device: {gaus_means.device}, shape: {gaus_means.shape}")
        print(f"DEBUG: gaus_opacities device: {gaus_opacities.device}, shape: {gaus_opacities.shape}")
        print(f"DEBUG: gaus_colors    device: {gaus_colors.device}, shape: {gaus_colors.shape}")
        print(f"DEBUG: gaus_covs      device: {gaus_covs.device}, shape: {gaus_covs.shape}")
        print(f"DEBUG: gaus_bitmaps   device: {gaus_bitmaps.device}, shape: {gaus_bitmaps.shape}")
        print("-------------------------------------------")
        # --- End of debugging code ---
 
        try:
            gaus_inv_covs = torch.linalg.inv(gaus_covs)
        except RuntimeError:
            continue

        # CUDA kernel launch parameters
        threads_per_block = (16, 16)
        blocks_x = (tile_size + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_y = (tile_size + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_x, blocks_y)

        gaus_means_numba     = cuda.as_cuda_array(gaus_means)
        gaus_inv_covs_numba  = cuda.as_cuda_array(gaus_inv_covs)
        gaus_opacities_numba = cuda.as_cuda_array(gaus_opacities)
        gaus_colors_numba    = cuda.as_cuda_array(gaus_colors)
        gaus_bitmaps_numba   = cuda.as_cuda_array(gaus_bitmaps)

        # image_buffer and alpha_buffer also need to be converted if they are PyTorch tensors
        image_buffer_numba = cuda.as_cuda_array(image_buffer)
        alpha_buffer_numba = cuda.as_cuda_array(alpha_buffer)

        # Kernel invocation
        rasterize_tile_kernel[blocks_per_grid, threads_per_block](
            image_buffer_numba, alpha_buffer_numba,
            tx, ty, tile_size,
            gaus_means_numba, gaus_inv_covs_numba, gaus_opacities_numba,
            gaus_colors_numba, gaus_bitmaps_numba, subtile_res
        )

    rendered_image = image_buffer.cpu().numpy()
    return rendered_image, metrics