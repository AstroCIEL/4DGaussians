# run_gscore_simulation.py (Modified Version)

import os
import argparse
from argparse import ArgumentParser
import numpy as np
from PIL import Image
import json
from dataclasses import dataclass, asdict
import torch # Added torch import
import torchvision.utils as vutils

# --- Import utilities from existing gaussian-splatting code ---
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
#from utils.system_utils import setup_logger
# ----------------------------------------------------

from gscore_simulator.simulator import GSCoreSimulator
# Import utils.py functions directly to allow passing the device argument
from gscore_simulator import utils as gscore_utils 

def tensor_to_builtin(o):
    if isinstance(o, torch.Tensor):
        if o.numel() == 1:
            return o.item()       # Scalar tensor -> number
        else:
            return o.tolist()     # Multi-dimensional tensor -> list
    raise TypeError(f"Type {o.__class__.__name__} not serializable")

def run_simulation(args):
    """Runs the entire GSCore simulation process and saves the results."""
    
    # Set specified device
    device = torch.device(args.device)
    
    gscore_config = {
        "tile_size": 16,
        "subtile_res": 4, 
        "obb_radius_scale": 3.0,
        "obb_test_ratio_threshold": 2.0,
        "gsu_chunk_size": 256,
        "device": args.device # Add device to configuration
    }
    
    #setup_logger()
    gaussians = GaussianModel(args.sh_degree, device=args.device)
    
    # Specify the device to be used by the Scene class
    args.data_device = args.device 
    setattr(args, 'images', 'images')
    setattr(args, 'depths', '')
    setattr(args, 'eval', True) # Set to True as we use test cameras
    setattr(args, 'train_test_exp', False)
    setattr(args, 'resolution', -1)
    device = args.data_device
    source_path = args.source_path
    scene_name = os.path.basename(os.path.normpath(source_path))

    print("\n=== Args passed to Scene ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================\n")

    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False, device=device)
    
    view_idx = args.view_index
    if view_idx >= len(scene.getTestCameras()):
        print(f"Error: View index {view_idx} is out of bounds for test set size {len(scene.getTestCameras())}")
        return
    camera = scene.getTestCameras()[view_idx]
    
    gt_image = None
    image_dir = os.path.join(args.source_path, getattr(args, 'images', 'images'))
    gt_image_path = os.path.join(image_dir, camera.image_name)
    print(f" \n gt image path: {gt_image_path}")
    

    # Run GSCore Simulator
    simulator = GSCoreSimulator(gscore_config)
    rendered_image, metrics = simulator.render(camera, gaussians)
    
    # Save images (gt image, rendered image)
    os.makedirs(args.output_dir, exist_ok=True)
    rendered_path = os.path.join(args.output_dir, f"gscore_render_{args.iteration}_{scene_name}_view{view_idx}.png")
    rendered_image = torch.from_numpy(rendered_image).to(device)
    vutils.save_image(rendered_image, rendered_path)

    # Load gt image
    try:
        
        gt_image_pil = Image.open(gt_image_path)
        gt_image = np.array(gt_image_pil, dtype=np.float32) / 255.0
        if gt_image.shape[2] == 4:
            gt_image = gt_image[:, :, :3]
        
        gt_image_pil = Image.open(gt_image_path).convert("RGB")
        W, H = gt_image_pil.size
        gt_image_pil = gt_image_pil.resize((rendered_image.shape[2], rendered_image.shape[1]), Image.BILINEAR)
        gt_image = np.array(gt_image_pil, dtype=np.float32) / 255.0
    except FileNotFoundError:
        print(f"Warning: Ground truth image not found. PSNR/LPIPS will not be calculated.")

    print(f"\nRendered image saved to: {rendered_path}")

    print("\n--- GSCore Simulation Report ---")
    # ... (Output section remains the same) ...
    print(f"Rendered Image Size: {rendered_image.shape[2]}x{rendered_image.shape[1]}")
    psnr = gscore_utils.calculate_psnr(rendered_image, gt_image)
    lpips = gscore_utils.calculate_lpips(rendered_image, gt_image)
    metrics.quality_metrics = {"PSNR": psnr, "LPIPS": lpips}
    if metrics.quality_metrics:
        print(f"PSNR: {metrics.quality_metrics['PSNR']:.4f} dB")
        print(f"LPIPS: {metrics.quality_metrics['LPIPS']:.4f}")
        
    print("\n--- Performance Metrics ---")
    print(f"Number of gaussians at {metrics.tile_coords}: {metrics.gaussians_per_tile}")
    print(f"Avg Gaussians Per Tile: {metrics.avg_gaussians_per_tile:.2f}")
    print(f"Max Gaussians Per Tile: {metrics.max_gaussians_per_tile}")
    print(f"MAC Per Tile: {metrics.macs_per_tile}")
    print(f"Tile Coords (example): {metrics.tile_coords}")
    print(f"[METRICS] Total pixel-blending operations: {metrics.total_blending_operations:,}")
    #print(f"Alpha Blending Operations: {metrics.alpha_blending_ops:,}")
    #print(f"Estimated MAC Operations: {metrics.mac_operations:,}")
  

    # Save metrics to a JSON file
    metrics_dir = "/home/hyoh/GSCore/gscore_renders/metrics"
    metrics_output_path = os.path.join(metrics_dir, f"gscore_metrics_{args.iteration}_{scene_name}_view{view_idx}.json")
    
    # Use asdict to convert RenderMetrics instance to a dictionary
    # Automatically converted according to the types of the dataclass fields
    metrics_to_save = asdict(metrics)

    # If gaussians_per_tile is large, the JSON file size may increase,
    # so you can decide whether to save this field as needed.
    # It is included here as an example.
    
    with open(metrics_output_path, "w") as f:
        json.dump(metrics_to_save, f, default=tensor_to_builtin, indent=4)
    print(f"Metrics saved to: {metrics_output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run GSCore simulation on a 3D Gaussian Splatting scene.")
    # ... (Existing arguments) ...
    parser.add_argument("--model_path", "-m", required=True, help="Path to the model directory (output of training).")
    parser.add_argument("--source_path", "-s", required=True, help="Path to the dataset source folder.")
    parser.add_argument("--iteration", "-i", default=30000, type=int, help="Iteration number of the model to load.")
    parser.add_argument("--output_dir", "-o", default="./gscore_renders/images", help="Directory to save rendered images.")
    parser.add_argument("--view_index", default=0, type=int, help="Index of the test camera view to render.")
    parser.add_argument("--sh_degree", default=3, type=int, help="SH degree of the Gaussian model.")

    # --- Added argument for specifying GPU device ---
    parser.add_argument("--device", default="cuda:0", help="Device to use for PyTorch operations (e.g., 'cuda:0', 'cuda:1', 'cpu').")
    
    args = parser.parse_args()
    args.white_background = False
    # data_device is set above, so removed here
    
    # Modified to pass iteration count directly when creating 'Scene' class
    # (Responsive to changes in the original codebase's Scene constructor)
    # This needs flexible adjustment depending on the version of the original code
    # setattr(args, 'load_iteration', args.iteration) 

    run_simulation(args)