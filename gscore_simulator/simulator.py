# gscore_simulator/simulator.py

import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
from gscore_simulator.culling import cull_and_convert
from gscore_simulator.sorting import hierarchical_sort_and_group
from gscore_simulator.rasterizer import rasterize_tiles
from gscore_simulator.utils import calculate_psnr, calculate_lpips
from gscore_simulator.structures import RenderMetrics


class GSCoreSimulator:
    """
    Class that simulates the entire pipeline of GSCore hardware accelerator.
    Sequentially calls each step of CCU, GSU, VRU and reports final results and performance.
    """
    def __init__(self, config: dict):
        """
        Initialize simulator

        Args:
            config (dict): Global configuration values for simulation
                           (e.g., tile_size, obb_test_ratio_threshold, etc.)
        """
        self.config = config
        self.device = self.config.get('device', 'cuda:0')
        print("GSCore Simulator initialized with config:")
        for key, value in config.items():
            print(f"  - {key}: {value}")


    def render(self, camera, gaussian_model):
        """
        Render 3D Gaussian model from given camera viewpoint.

        Args:
            camera: Camera object for rendering (from gaussian-splatting)
            gaussian_model: Trained 3D Gaussian model (from gaussian-splatting)
            gt_image (np.ndarray, optional): Ground-Truth image for quality evaluation.

        Returns:
            tuple: (rendered image, performance metrics object)
        """
        # Keep original camera as is, create copy with half resolution
        #torch.cuda.empty_cache()
        camera_half = deepcopy(camera)
        #camera_half.image_width //= 2
        #camera_half.image_height //= 2
        metrics = RenderMetrics()

        # --- CCU (Culling and Conversion Unit) ---
        start_time = time.time()
        culled_gaussians, G_list, metrics = cull_and_convert(gaussian_model, camera_half, self.config)
        ccu_time = time.time() - start_time
        print(f"CCU finished in {ccu_time:.2f}s.")

        # --- GSU (Gaussian Sorting Unit) ---
        start_time = time.time()
        tile_to_chunks, metrics = hierarchical_sort_and_group(G_list, self.config, metrics)
        gsu_time = time.time() - start_time
        print(f"GSU finished in {gsu_time:.2f}s.")
        
        # --- VRU (Volume Rendering Unit) ---
        start_time = time.time()
        rendered_image, metrics = rasterize_tiles(tile_to_chunks, culled_gaussians, camera_half, self.config, self.device, metrics)
        vru_time = time.time() - start_time
        print(f"VRU finished in {vru_time:.2f}s.")

        return rendered_image, metrics