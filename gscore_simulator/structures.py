# gscore_simulator/structures.py

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class RenderMetrics:
    """Data class that stores performance metrics during rendering process"""
    gaussians_per_tile: List[int] = None
    # Reference tile starting coordinates
    tile_coords: List[Tuple[int, int]] = None
    tile_idxs: List[int] = None
    mac_per_tile: List[int] = 0
    avg_gaussians_per_tile: int = 0
    max_gaussians_per_tile: int = 0
    macs_per_tile: int = 0
    avg_saved_gaussians: float = 0.0
    avg_saved_rate: float = 0.0
    avg_calculated_gaussians: float = 0.0
    first_saved_gaussians: float = 0.0
    first_saved_rate: float = 0.0
    first_calculated_gaussians: float = 0.0
    first_terminated_coords: Tuple[int, int] = None
    total_blending_operations: int = 0
    # Final quality metrics like PSNR, LPIPS
    quality_metrics: dict = field(default_factory=dict)

@dataclass
class Gaussian2D:
    """
    Class that holds information after 3D Gaussian is projected onto 2D image plane.
    Used in hierarchical_sort_and_group() for creation, sorting, and grouping.
    """
    # Index of original 3D Gaussian
    source_id: int

    # Center coordinates on 2D plane (x, y)
    mean: torch.Tensor            # shape: (2,)

    # 2D covariance matrix (form: cov2d)
    cov: torch.Tensor             # shape: (2,2) or upper triangular elements, etc.

    # Depth value (used for sorting)
    depth: float

    # Opacity
    opacity: float

    # Spherical Harmonics coefficients (used for color calculation)
    color_precomp: torch.Tensor   # shape: (...,)

    # Subtile-level bitmap for each intersecting tile
    # { tile_id: {"bitmap": tensor(shape=(S,)), "start": (x0,y0)} }
    tiles: Dict[int, Dict[str, any]]

    def __lt__(self, other: "Gaussian2D"):
        """Sort in ascending order by depth"""
        return self.depth < other.depth