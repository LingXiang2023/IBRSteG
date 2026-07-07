import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from stegamodels.stegotool import *


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=10, out_channels=5):  # 修改输入输出通道数
        super(Encoder, self).__init__()

        self.n_channels = in_channels
        self.out_channels = out_channels

        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.down6 = Down(512, 512)
        self.down7 = Down(512, 512)
        
        self.up1 = Up(512, 512)
        self.up2 = Up(1024, 512)
        self.up3 = Up(1024, 512)
        self.up4 = Up(1024, 256)
        self.up5 = Up(512, 128)
        self.up6 = Up(256, 64)

        self.outlayer = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        )

    def forward(self, x):
        x1 = self.down1(x)      # 64   * 64 *64
        x2 = self.down2(x1)     # 128  * 32 *32
        x3 = self.down3(x2)     # 256  * 16 *16
        x4 = self.down4(x3)     # 512  * 8  *8
        x5 = self.down5(x4)     # 512  * 4  *4     
        x6 = self.down6(x5)     # 512  * 2  *2
        x7 = self.down7(x6)     # 512  * 1  *1

        x = self.up1(x7, x6)    #1024  * 2  *2  
        x = self.up2(x, x5)     #1024  * 4  *4
        x = self.up3(x, x4)     #1024  * 8  *8
        x = self.up4(x, x3)     #512   * 16 *16
        x = self.up5(x, x2)     #256   * 32 *32
        x = self.up6(x, x1)     #128   * 64 *64
        stego = self.outlayer(x)# 5    * 128*128
        
        return stego


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()

        self.dec_net = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=100, kernel_size=3, stride=1, padding=1),  # 修改输入通道
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=5, kernel_size=1, stride=1, padding=0),  # 修改输出通道
            nn.Sigmoid()
        )

    def forward(self, stego):
        secret_rev = self.dec_net(stego)
        return secret_rev


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Sigmoid):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block
        self.output_layer= MultiHeadNetwork(head_dim=32)
    def forward(self,cover,secret):
        input = torch.cat((secret, cover), dim=1)
        output= self.model(input)
        main_output,special_output=self.output_layer(output)
        return main_output,special_output

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Sigmoid):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class RevealNet(nn.Module):
    def __init__(self, nc=5, nhf=64, output_function=nn.Sigmoid):  # 修改输入输出通道数
        super(RevealNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf*4),
            nn.ReLU(True),
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nc, 3, 1, 1),  # 输出通道改为5
            output_function()
        )

    def forward(self, input):
        output = self.main(input)
        return output


class MultiHeadNetwork(nn.Module):
    def __init__(self, head_dim, rgb_dim=3, rot_dim=4, opacity_dim=1, depth_dim=1):
        """
        多头网络，预测3D高斯泼溅的各个参数
        
        Args:
            head_dim: 主干输出的特征维度
            rgb_dim: RGB颜色通道数 (默认3)
            rot_dim: 旋转参数通道数 (默认4，四元数)
            opacity_dim: 不透明度通道数 (默认1)
            depth_dim: 深度通道数 (默认1)
        """
        super(MultiHeadNetwork, self).__init__()
        
        self.head_dim = head_dim
        self.rgb_dim = rgb_dim
        self.rot_dim = rot_dim
        self.opacity_dim = opacity_dim
        self.depth_dim = depth_dim
        
        # RGB颜色头 (使用Sigmoid限制到[0,1]范围)
        self.rgb_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, self.rgb_dim, kernel_size=1),
            nn.Sigmoid()  # 将RGB值限制在[0,1]范围内
        )
        
        # 旋转参数头 (四元数表示，无特殊激活函数)
        self.rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, self.rot_dim, kernel_size=1),
        )
        
        # 不透明度头 (使用Sigmoid限制到[0,1]范围)
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, self.opacity_dim, kernel_size=1),
            nn.Sigmoid()  # 不透明度在[0,1]范围内
        )
        
        # 深度头 (使用Tanh限制到[-1,1]范围，可根据需要调整)
        self.depth_head_resi= nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, self.depth_dim, kernel_size=1),
            nn.Tanh()  # 深度值归一化到[-1,1]范围
        )
       
        self.depth_head_coarse = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, self.depth_dim, kernel_size=1),
            nn.Sigmoid()   # 深度值归一化到[-1,1]范围
        )
        
        # 总输出通道数
        self.total_output_dim = rgb_dim + rot_dim + opacity_dim + depth_dim
        
    def forward(self, x):
       
       
        # 分别通过各个头
        rgb_out = self.rgb_head(x)      # [B, 3, H, W]
        rot_out = self.rot_head(x)      # [B, 4, H, W]
        opacity_out = self.opacity_head(x)  # [B, 1, H, W]
        depth_out = self.depth_head_coarse(x)  # [B, 1, H, W]
        depth_resi=self.depth_head_resi(x)*0.5
        # 在通道维度上拼接
        output = torch.cat([depth_out,rgb_out, rot_out, opacity_out], dim=1)
        
        return output,depth_resi
#wE use it for gaussian
class GaussianAttributesSteganographer(nn.Module):
    def __init__(self) -> None:
        super(GaussianAttributesSteganographer, self).__init__()

        self.encoder = UnetGenerator(input_nc=18, output_nc=32, num_downs=7, output_function=nn.Sigmoid)  # 调整输入输出通道
        self.decoder = RevealNet(nc=9)  # 调整输入通道
    def load_pretrained(self, load_path, device, strict=True):
        # 加载预训练权重
        ckpt = torch.load(load_path, map_location=device)
        
        # 获取当前模型的状态字典
        model_state_dict = self.state_dict()
        
        # 筛选出预训练权重中存在的键（排除finer部分）
        pretrained_state_dict = {}
        for k, v in ckpt['conet'].items():
            if k in model_state_dict and not k.startswith('finer.'):
                pretrained_state_dict[k] = v
        
        # 加载权重
        model_state_dict.update(pretrained_state_dict)
        self.load_state_dict(model_state_dict, strict=strict)
        '''
        # 冻结encoder和decoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        '''
    def forward(self,coverdata, secretdata, mode='train'):

        outcome=[]
        losses=[]
        for view in ['lmain', 'rmain']:
            #这里都没梯度
            gaussiancover=normalize_gaussian(get_gaussian(coverdata,view),view)
            gaussiansecret=normalize_gaussian(get_gaussian(secretdata,view),view)
            #
            gaussian,losses_stego=twod_encode(self.encoder,gaussiancover,gaussiansecret)
            
            #print(gaussian_psnr(gaussian,gaussiancover))
            outcome.append(denormalize_gaussian(gaussian,view))
            losses.append(losses_stego)
        data_stego= make_data(coverdata,outcome) 
        #print(data_stego['lmain']['img'].requires_grad) 
        #目前，这个data_stego应该带上了梯度
        #data_cover cannot reuse!!!
        outcome=[]
        for view in ['lmain', 'rmain']:
            #这里都没梯度

            gaussianrev=normalize_gaussian(get_gaussian(data_stego,view),view)
            gaussiansecret=normalize_gaussian(get_gaussian(secretdata,view),view)
            gaussian,losses_rev=twod_decode(self.decoder,gaussianrev,gaussiansecret)
            #print(gaussian_psnr(gaussian,gaussiansecret))
            
            outcome.append(denormalize_gaussian(gaussian,view))
            losses.append(losses_rev)
        data_rev=make_data(secretdata,outcome)
        if mode=='train':
            return data_stego, data_rev,losses
        else:
            return data_stego, data_rev


# Backward-compatible alias for checkpoints and older scripts.
Conet = GaussianAttributesSteganographer
