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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
def ssim_masked(img1, img2, mask, window_size=11, size_average=True, eps=1e-8):
    '''
    img1, img2: (N, C, H, W), float, [0,1]
    mask: (N, 1, H, W) or (N, H, W), bool or float
    '''
    N, C, H, W = img1.shape
    
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (N,1,H,W)
    mask = mask.float()

    window = create_window(window_size, C).to(img1.device).type_as(img1)

    # 用mask加权卷积获得masked均值
    def masked_conv(img):
        img_masked = img * mask
        mu = F.conv2d(img_masked, window, padding=window_size // 2, groups=C)
        mask_sum = F.conv2d(mask, window, padding=window_size // 2, groups=1)
        mu = mu / (mask_sum + eps)
        return mu, mask_sum

    mu1, mask_sum = masked_conv(img1)
    mu2, _ = masked_conv(img2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1*mask, window, padding=window_size // 2, groups=C) / (mask_sum + eps) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2*mask, window, padding=window_size // 2, groups=C) / (mask_sum + eps) - mu2_sq
    sigma12  = F.conv2d(img1*img2*mask, window, padding=window_size // 2, groups=C) / (mask_sum + eps) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 只统计mask>0的区域
    if size_average:
        mask_valid = (mask_sum > eps).float()
        ssim_valid = ssim_map * mask_valid
        ssim_mean = ssim_valid.sum() / (mask_valid.sum() + eps)
        return ssim_mean
    else:
        mask_valid = (mask_sum > eps).float()
        ssim_valid = ssim_map * mask_valid
        ssim_mean = ssim_valid.sum(dim=[2,3]) / (mask_valid.sum(dim=[2,3]) + eps)
        return ssim_mean  # (N, C)


