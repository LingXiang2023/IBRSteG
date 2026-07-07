#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def get_valid_mask(img_pred, bg_color=[0, 0, 0]):
    # img_pred: (B, H, W, 3) or (B, 3, H, W), float, [0,1]
    bg_color = torch.tensor(bg_color, dtype=img_pred.dtype, device=img_pred.device)
    
    if img_pred.ndim == 4:
        # Batch processing
        if img_pred.shape[1] == 3:  # NCHW format
            bg_color = bg_color.view(1, 3, 1, 1)
            mask = (img_pred != bg_color).any(dim=1)
        else:  # NHWC format
            bg_color = bg_color.view(1, 1, 1, 3)
            mask = (img_pred != bg_color).any(dim=-1)
    else:
        # Single image (no batch dimension)
        if img_pred.shape[0] == 3:  # CHW format
            bg_color = bg_color.view(3, 1, 1)
            mask = (img_pred != bg_color).any(dim=0)
        else:  # HWC format
            bg_color = bg_color.view(1, 1, 3)
            mask = (img_pred != bg_color).any(dim=-1)
    
    return mask

def masked_psnr(pred, gt, mask):
    # pred, gt: (B, 3, H, W), mask: (B, H, W), bool
    # Ensure mask has proper dimensions for broadcasting
    mask = mask.unsqueeze(1)  # (B, 1, H, W)
    
    # Calculate MSE only on masked pixels
    squared_error = (pred - gt) ** 2
    masked_squared_error = squared_error * mask
    
    # Compute MSE per image in the batch
    sum_mse = masked_squared_error.view(pred.shape[0], -1).sum(dim=1)
    valid_pixels = mask.view(pred.shape[0], -1).sum(dim=1)
    
    # Avoid division by zero
    mse = sum_mse / (valid_pixels + 1e-8)
    
    # Compute PSNR for each image in the batch
    psnr = -10 * torch.log10(mse + 1e-8)
    return psnr
