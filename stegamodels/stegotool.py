import torch.nn.functional as F
import torch
import numpy as np
#make sure all the operation is reversible
import os
import torch
def alphaloss(network_output, gt,alpha):
    weight= (alpha>0.8).float()
    return torch.abs((network_output - gt)*weight).mean()/weight.mean()


def flip_boolean_tensor(mask, p):
    """
    随机将布尔张量中比例为 p 的元素取反。
    
    参数:
    mask (torch.Tensor): 输入的布尔型张量
    p (float): 取反的比例 (0 <= p <= 1)
    """
    if not (0 <= p <= 1):
        raise ValueError("比例 p 必须在 0 到 1 之间")
    
    # 1. 生成一个与 mask 形状相同的随机浮点张量
    # 2. 比较随机值与 p，得到一个布尔掩码（选中比例约为 p 的元素）
    flip_mask = torch.rand(mask.shape, device=mask.device) < p
    
    # 3. 使用异或运算 (^)：True ^ True = False, False ^ True = True
    # 这会直接翻转 flip_mask 中为 True 的位置
    return mask ^ flip_mask

# 针对你的变量调用示例
# gaussian1_0_flipped = flip_boolean_tensor(gaussian1[0], p=0.1)

def mask_crop(x, mask):

    B, C, W, _ = x.shape
    
    # 将掩码重塑为空间维度 [1, W, W]
    spatial_mask = mask.view(1, W, W)
    
    # 扩展掩码以匹配输入张量的形状 [B, C, W, W]
    expanded_mask = spatial_mask.expand(B, C, W, W)
    
    # 使用掩码进行裁剪，保持梯度流
    result = torch.zeros_like(x)
    result[expanded_mask] = x[expanded_mask]
    
    return result

def mask_crop_gaussian(gaussian):
    bool_mask=gaussian[0]
    for i in [1,2,3,5]:
        gaussian[i]=mask_crop(gaussian[i],bool_mask)
    return gaussian
    

def normalize_gaussian(gaussian,view):
    valid, depth, rgb, rot, scale, opacity = gaussian
    
    # 归一化到[0,1]范围
    depth_normalized = depth*100/255
    rgb_normalized = (rgb + 1.0) / 2.0
    scale_normalized = scale
    opacity_normalized = opacity  # 已经在[0,1]范围内，无需处理
    
    return [valid, depth_normalized, rgb_normalized, rot, scale_normalized, opacity_normalized]

def denormalize_gaussian(gaussian_normalized,view):
    valid, depth_norm, rgb_norm, rot_norm, scale_norm, opacity_norm = gaussian_normalized
    
    # 从[0,1]范围反归一化到原始范围
    depth = depth_norm * 2.55
    rgb = rgb_norm * 2.0 - 1.0
    scale = scale_norm
    opacity = opacity_norm  # 保持在[0,1]范围内
    
    return [valid, depth, rgb, rot_norm, scale, opacity]
def get_gaussian(data,view):
    assert view=='lmain' or view=='rmain'
    return [data[view]['pts_valid'],data[view]['depth'],data[view]['img'],data[view]['rot_maps'],data[view]['scale_maps'],data[view]['opacity_maps']]
def twod_encode(encoder, gaussian1, gaussian2):
    gaussian1 = mask_crop_gaussian(gaussian1)
    gaussian2 = mask_crop_gaussian(gaussian2)
    depth1 = gaussian1[1]
    rgb1 = gaussian1[2]
    rot1 = gaussian1[3]
    scale1 = gaussian1[4]
    opacity1 = gaussian1[5]
    
    depth2 = gaussian2[1]
    rgb2 = gaussian2[2]
    rot2 = gaussian2[3]
    scale2 = gaussian2[4]
    opacity2 = gaussian2[5]
    
    scale_stego = torch.ones_like(scale1) * 0.002
    
    outcome_stego,depth_resi = encoder(
        torch.cat([depth1, rgb1, rot1, opacity1], dim=1),
        torch.cat([depth2, rgb2, rot2, opacity2], dim=1)
    )
    
    outcome_gaussian = [
        gaussian1[0], 
        depth1+depth_resi,
        outcome_stego[:, [1, 2, 3], :, :], 
        torch.nn.functional.normalize(outcome_stego[:, [4, 5, 6, 7], :, :], dim=1), 
        scale_stego, 
        outcome_stego[:, 8, :, :].unsqueeze(1)
    ]
    
    outcome_gaussian = mask_crop_gaussian(outcome_gaussian)
    # 计算损失（仅在训练模式下）
   
        # RGB损失
    rgb_loss = F.l1_loss(outcome_gaussian[2], rgb1)*0.5
        # Depth损失
    depth_loss =torch.abs(depth_resi).mean()*0.5
        # Alpha/Opacity损失
    alpha_loss = F.l1_loss(outcome_gaussian[5], opacity1)*0.5
        
    losses = (rgb_loss, depth_loss, alpha_loss)
    outcome_gaussian = mask_crop_gaussian(outcome_gaussian)
    return outcome_gaussian, losses

    
def twod_decode(decoder, stego_gaussian, gaussian1=None): 
    #stego_gaussian[0] = flip_boolean_tensor(stego_gaussian[0], p=0.2)
    stego_gaussian=mask_crop_gaussian(stego_gaussian)
    gaussian1=mask_crop_gaussian(gaussian1)
    valid_stego = gaussian1[0]
    depth_stego = stego_gaussian[1]
    rgb_stego = stego_gaussian[2]
    rot_stego = stego_gaussian[3]
    scale_stego = stego_gaussian[4]
    opacity_stego = stego_gaussian[5]
    
    outcome_recover = decoder(torch.cat([depth_stego, rgb_stego, rot_stego, opacity_stego], dim=1))
    
    scale_recover = torch.ones_like(scale_stego) * 0.002
    
    gaussian_recover = [
        gaussian1[0],
        outcome_recover[:, 0, :, :].unsqueeze(1), 
        outcome_recover[:, [1, 2, 3], :, :], 
        torch.nn.functional.normalize(outcome_recover[:, [4, 5, 6, 7], :, :], dim=1), 
        scale_stego, 
        outcome_recover[:, 8, :, :].unsqueeze(1)
    ]
    #try
    
    gaussian_recover=mask_crop_gaussian(gaussian_recover)
    
    #
    # 计算损失（仅在训练模式下且提供了gaussian1）
        # RGB损失
    rgb_loss = F.l1_loss(gaussian_recover[2], gaussian1[2])*0.5
        # Depth损失
    depth_loss = F.l1_loss(gaussian_recover[1], gaussian1[1])*0.5
        # Alpha/Opacity损失
    alpha_loss = F.l1_loss(gaussian_recover[5], gaussian1[5])*0.5
        
    losses = (rgb_loss, depth_loss, alpha_loss)
    
    return gaussian_recover, losses
#并没有复制pts
def make_data(data_old,gaussianlist):
    for idx,view in enumerate(['lmain','rmain']):
        data_old[view]['depth']=gaussianlist[idx][1]
        data_old[view]['img']=gaussianlist[idx][2]
        data_old[view]['rot_maps']=gaussianlist[idx][3]
        data_old[view]['scale_maps']=gaussianlist[idx][4]
        data_old[view]['opacity_maps']=gaussianlist[idx][5]
        
    return data_old
    

def statistic_gaussian(gaussian, save_dir="statistics", prefix="gaussian"):
    """
    计算高斯参数的统计学信息并保存直方图
    
    Args:
        gaussian: 高斯参数列表 [valid, depth, rgb, rot, scale, opacity]
        save_dir: 保存直方图的目录
        prefix: 文件前缀
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 解包高斯参数
    valid, depth, rgb, rot, scale, opacity = gaussian
    
    # 确保所有张量都在CPU上
    valid = valid.detach().cpu()
    depth = depth.detach().cpu()
    rgb = rgb.detach().cpu()
    rot = rot.detach().cpu()
    opacity = opacity.detach().cpu()
    
    # 获取有效区域的掩码 (B, W, W) -> (B, 1, W, W) 用于广播
    B, C, W, _ = depth.shape
    
    # 将掩码重塑为空间维度 [1, W, W]
    valid= valid.view(1, W, W)
    valid_mask = valid.unsqueeze(1).expand_as(depth)
    
    # 只计算有效区域的值
    def get_valid_values(tensor, mask):
        mask_expanded = mask.expand_as(tensor)  # 或者 mask.repeat(1, 3, 1, 1)
        return tensor[mask_expanded > 0.5]
    
    # 计算每个参数的统计信息
    stats = {}
    
    print(f"=== Gaussian Statistics for {prefix} ===")
    
    # Depth 统计 (单通道)
    valid_depth = get_valid_values(depth, valid_mask)
    if len(valid_depth) > 0:
        depth_mean = valid_depth.mean().item()
        depth_std = valid_depth.std().item()
        stats['depth'] = {'mean': depth_mean, 'std': depth_std, 'values': valid_depth}
        print(f"Depth - Mean: {depth_mean:.4f}, Std: {depth_std:.4f}")
    
    # RGB 统计 (多通道取平均)
    valid_rgb = get_valid_values(rgb, valid_mask)
    if len(valid_rgb) > 0:
        # 计算所有通道的平均值
        rgb_global_mean = valid_rgb.mean().item()
        rgb_global_std = valid_rgb.std().item()
        stats['rgb'] = {'mean': rgb_global_mean, 'std': rgb_global_std, 'values': valid_rgb}
        print(f"RGB (Global) - Mean: {rgb_global_mean:.4f}, Std: {rgb_global_std:.4f}")
    
    # Rotation 统计 (多通道取平均)
    valid_rot = get_valid_values(rot, valid_mask)
    if len(valid_rot) > 0:
        # 计算所有通道的平均值
        rot_global_mean = valid_rot.mean().item()
        rot_global_std = valid_rot.std().item()
        stats['rot'] = {'mean': rot_global_mean, 'std': rot_global_std, 'values': valid_rot}
        print(f"Rotation (Global) - Mean: {rot_global_mean:.4f}, Std: {rot_global_std:.4f}")
    
    # Opacity 统计 (单通道)
    valid_opacity = get_valid_values(opacity, valid_mask)
    if len(valid_opacity) > 0:
        opacity_mean = valid_opacity.mean().item()
        opacity_std = valid_opacity.std().item()
        stats['opacity'] = {'mean': opacity_mean, 'std': opacity_std, 'values': valid_opacity}
        print(f"Opacity - Mean: {opacity_mean:.4f}, Std: {opacity_std:.4f}")
    
    print("=" * 50)
    
    # 绘制直方图
    plot_histograms(stats, save_dir, prefix)
    
    return stats

def plot_histograms(stats, save_dir, prefix):
    """绘制所有参数的直方图"""
    import matplotlib.pyplot as plt
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 12})
    
    # Depth 直方图
    if 'depth' in stats:
        plt.figure(figsize=(12, 8))
        n, bins, patches = plt.hist(stats['depth']['values'].numpy(), bins=50, alpha=0.8, 
                                   color='skyblue', edgecolor='navy', linewidth=0.5)
        
        # 添加均值和标准差线
        mean_line = plt.axvline(stats['depth']['mean'], color='red', linestyle='--', 
                               linewidth=2.5, label=f'Mean: {stats["depth"]["mean"]:.4f}')

        plt.title(f'Depth Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Depth Value', fontsize=14, fontweight='bold')
        plt.ylabel('Frequency', fontsize=14, fontweight='bold')
        
        # 美化图例
        legend = plt.legend(loc='upper right', fontsize=12, frameon=True, 
                           fancybox=True, shadow=True, framealpha=0.9)
        legend.get_frame().set_facecolor('lightgray')
        
        plt.grid(True, alpha=0.4, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_depth_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # RGB 直方图 (全局)
    if 'rgb' in stats:
        plt.figure(figsize=(12, 8))
        n, bins, patches = plt.hist(stats['rgb']['values'].numpy(), bins=50, alpha=0.8, 
                                   color='lightgreen', edgecolor='darkgreen', linewidth=0.5)
        
        mean_line = plt.axvline(stats['rgb']['mean'], color='red', linestyle='--', 
                               linewidth=2.5, label=f'Mean: {stats["rgb"]["mean"]:.4f}')
 
        
        plt.title(f'RGB Global Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('RGB Value', fontsize=14, fontweight='bold')
        plt.ylabel('Frequency', fontsize=14, fontweight='bold')
        
        legend = plt.legend(loc='upper right', fontsize=12, frameon=True, 
                           fancybox=True, shadow=True, framealpha=0.9)
        legend.get_frame().set_facecolor('lightgray')
        
        plt.grid(True, alpha=0.4, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_rgb_global_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Rotation 直方图 (全局)
    if 'rot' in stats:
        plt.figure(figsize=(12, 8))
        n, bins, patches = plt.hist(stats['rot']['values'].numpy(), bins=50, alpha=0.8, 
                                   color='plum', edgecolor='purple', linewidth=0.5)
        
        mean_line = plt.axvline(stats['rot']['mean'], color='red', linestyle='--', 
                               linewidth=2.5, label=f'Mean: {stats["rot"]["mean"]:.4f}')

        
        plt.title(f'Rotation Global Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Rotation Value', fontsize=14, fontweight='bold')
        plt.ylabel('Frequency', fontsize=14, fontweight='bold')
        
        legend = plt.legend(loc='upper right', fontsize=12, frameon=True, 
                           fancybox=True, shadow=True, framealpha=0.9)
        legend.get_frame().set_facecolor('lightgray')
        
        plt.grid(True, alpha=0.4, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_rotation_global_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Opacity 直方图
    if 'opacity' in stats:
        plt.figure(figsize=(12, 8))
        n, bins, patches = plt.hist(stats['opacity']['values'].numpy(), bins=50, alpha=0.8, 
                                   color='gold', edgecolor='darkorange', linewidth=0.5)
        
        mean_line = plt.axvline(stats['opacity']['mean'], color='red', linestyle='--', 
                               linewidth=2.5, label=f'Mean: {stats["opacity"]["mean"]:.4f}')
 
        
        plt.title(f'Opacity Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Opacity Value', fontsize=14, fontweight='bold')
        plt.ylabel('Frequency', fontsize=14, fontweight='bold')
        
        legend = plt.legend(loc='upper right', fontsize=12, frameon=True, 
                           fancybox=True, shadow=True, framealpha=0.9)
        legend.get_frame().set_facecolor('lightgray')
        
        plt.grid(True, alpha=0.4, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_opacity_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
# 使用示例

    
def gaussian_psnr(gaussian_pred, gaussian):
    # 解包预测和真实值
    _, depth_pred, rgb_pred, rot_pred, _, opacity_pred = gaussian_pred
    _, depth, rgb, rot, _, opacity = gaussian
    
    # 计算每个特征的MSE
    def compute_mse(pred, target):
        return F.mse_loss(pred, target, reduction='mean')
    
    # 计算深度PSNR
    depth_mse = compute_mse(depth_pred, depth)
    depth_psnr = 10 * torch.log10(1.0 / depth_mse)
    
    # 计算RGB PSNR
    rgb_mse = compute_mse(rgb_pred, rgb)
    rgb_psnr = 10 * torch.log10(1.0 / rgb_mse)
    
    # 计算旋转PSNR
    rot_mse = compute_mse(rot_pred, rot)
    rot_psnr = 10 * torch.log10(1.0 / rot_mse)
    
    # 计算透明度PSNR
    opacity_mse = compute_mse(opacity_pred, opacity)
    opacity_psnr = 10 * torch.log10(1.0 / opacity_mse)
    
    return depth_psnr, rgb_psnr, rot_psnr, opacity_psnr   
    

