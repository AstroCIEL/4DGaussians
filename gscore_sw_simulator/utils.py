# gscore_simulator/utils.py (Modified Version)

import torch
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity

def calculate_psnr(img1, img2, device="cpu"):
    """Calculate PSNR between two images on the specified device."""
    # Move metric calculation object to specified device
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    
       
    # Process img1 (rendered_image)
    if isinstance(img1, np.ndarray):
        img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
    elif isinstance(img1, torch.Tensor):
        if img1.ndim == 3:
            img1_t = img1.float()
        elif img1.ndim == 4:
            img1_t = img1.float()
        else:
            raise ValueError("img1 shape must be (H, W, C) or (1, C, H, W)")
    else:
        raise TypeError("img1 must be np.ndarray or torch.Tensor")

    # Process img2
    if isinstance(img2, np.ndarray):
        img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()
    elif isinstance(img2, torch.Tensor):
        img2_t = img2.float()
    else:
        raise TypeError("img2 must be np.ndarray or torch.Tensor")
    
    # Move to device (important!)
    img1_t = img1_t.to(device)
    img2_t = img2_t.to(device)
    
    return psnr_metric(img1_t, img2_t).item()

def calculate_lpips(img1, img2, device="cpu", net_type='alex'):
    """Calculate LPIPS between two images on the specified device."""
    print("img1 range:", img1.min().item(), img1.max().item())
    print("img2 range:", img2.min().item(), img2.max().item())

    
    # Move metric calculation object to specified device
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=True).to(device)
    
       
    # Process img1 (rendered_image)
    if isinstance(img1, np.ndarray):
        img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
    elif isinstance(img1, torch.Tensor):
        if img1.ndim == 3:
            img1_t = img1.permute(2, 0, 1).unsqueeze(0).float()
            img1_t = img1.unsqueeze(0).float()
        elif img1.ndim == 4:
            img1_t = img1.float()
        else:
            raise ValueError("img1 shape must be (H, W, C) or (1, C, H, W)")
    else:
        raise TypeError("img1 must be np.ndarray or torch.Tensor")

    # Process img2
    if isinstance(img2, np.ndarray):
        img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()
    elif isinstance(img2, torch.Tensor):
        if img2.dim()== 3:
            img2_t = img2.unsqueeze(0).float()
        else:
            img2_t = img2.float()
    else:
        raise TypeError("img2 must be np.ndarray or torch.Tensor")
    
    # Move to device (important!)
    img1_t = img1_t.to(device)
    img2_t = img2_t.to(device)
    
    return lpips_metric(img1_t, img2_t).item()

# ... (get_view_matrix, get_projection_matrix functions remain unchanged) ...
def get_view_matrix(camera):
    return camera.world_view_transform


def get_projection_matrix(camera):
    return camera.full_proj_transform
