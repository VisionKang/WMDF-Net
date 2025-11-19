import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_module import *


class HinResBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x+resi


class PatchUnEmbed(nn.Module):

    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter

    def forward(self, x,x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size  # patch_size=1, stride=1, in_chans=32, embed_dim=32
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):

        # （b,c,h,w)->(b,c*s*p,h//s,w//s)
        # (b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x



# 预融合，细节增强 Detail information enhancement
class Dense_Block(nn.Module):
    def __init__(self):
        super(Dense_Block, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(5, 16, 3, 1, 1),
            nn.ReLU(True),
            nn.PixelShuffle(2),
            nn.Conv2d(4, 16, 3, 1, 1),
            nn.ReLU(True),
        )
        self.layers = nn.ModuleDict({
            'DenseConv1': nn.Conv2d(16, 16, 3, 1, 1),
            'DenseConv2': nn.Conv2d(32, 16, 3, 1, 1),
            'DenseConv3': nn.Conv2d(48, 16, 3, 1, 1),
        })
        self.layersR = nn.ModuleDict({
            'Resnet1': nn.Conv2d(16, 16, 3, 1, 1),
            'Resnet2': nn.Conv2d(16, 16, 3, 1, 1),
            'Resnet3': nn.Conv2d(16, 16, 3, 1, 1),
        })
        self.postConv = nn.Sequential(
            nn.Conv2d(80, 16, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(16, 4, 3, 1, 1),
        )

    def forward(self, pan, ms):
        # print("预融合，i细节增强 pan = {},ms = {}".format(pan.shape,ms.shape))
        input = torch.cat([ms, pan], 1)
        x = self.layer1(input)  # 16
        for i in range(len(self.layers)):
            out1 = self.layers['DenseConv'+str(i+1)](x)
            x = torch.cat([x, out1], 1)
        out1 = x  # 64

        y = self.layer1(input)  # 16
        out_1 = self.layersR['Resnet1'](y)
        out_2 = self.layersR['Resnet2'](out_1)
        out_3 = self.layersR['Resnet3'](y+out_2)
        out2 = out_3
        out = torch.cat([out1, out2], 1)
        out = self.postConv(out)

        return out


# def updown(x, size, mode='bicubic'):
#     out = F.interpolate(x, size=size, mode=mode, align_corners=True)
#     return out

def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)
# class Detail_information_enhancement(nn.Module):
#     def __init__(self):
#         super(Detail_information_enhancement, self).__init__()
#
#         self.up1 = Dense_Block()
#         self.up2 = Dense_Block()
#
#
#     def forward(self, pan, lr):
#         _, N, H, W = lr.shape
#         pan_4 = updown(pan, (H, W))
#         pan_2 = updown(pan, (H * 2, W * 2))
#         ms_2 = updown(lr, (H*2, W*2))
#         ms_4 = updown(lr, (H*4, W*4))
#         pan_blur = kornia.filters.GaussianBlur2d((11, 11), (1, 1))(pan)
#         pan_2_blur = kornia.filters.GaussianBlur2d((11, 11), (1, 1))(pan_2)
#         pan_hp = pan - pan_blur
#         pan_2_hp = pan_2 - pan_2_blur
#
#         lr_2 = self.up1(pan_4, lr) + ms_2 + pan_2_hp
#         lr_u = self.up2(pan_2, lr_2) + ms_4 + pan_hp
#         return lr_u

class Freprocess(nn.Module):
    def __init__(self, channels):
        super(Freprocess, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, msf, panf):

        _, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf), norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf), norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)
        amp_fuse = self.amp_fuse(torch.cat([msF_amp,panF_amp],1))
        pha_fuse = self.pha_fuse(torch.cat([msF_pha,panF_pha],1))

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)


class Multi_Scale_Feature_Extraction(nn.Module):        # 输入(1,16,128,128)
    def __init__(self, relu_slope=0.2, use_HIN=True, out_size=8):
        super(Multi_Scale_Feature_Extraction, self).__init__()
        self.conv_1 = nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN
        self.identity = nn.Sequential(
            nn.Conv2d(4, 4, 1, 1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=False)
        )
        self.Con1X1 = nn.Sequential(
            nn.Conv2d(8, 6, 1, 1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=False),
            nn.Conv2d(6, 3, 1, 1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=False)
        )
        self.Con3X3 = nn.Sequential(
            nn.Conv2d(8, 6, 3, 1, 1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=False),
            nn.Conv2d(6, 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=False)
        )
        self.Con5X5 = nn.Sequential(
            nn.Conv2d(8, 6, 5, 1, 2, bias=True),
            nn.LeakyReLU(relu_slope, inplace=False),
            nn.Conv2d(6, 3, 5, 1, 2, bias=True),
            nn.LeakyReLU(relu_slope, inplace=False)
        )

    def forward(self, X):
        X_1, X_2 = torch.chunk(X, 2, dim=1)
        y = self.relu_1(self.conv_1(X_2))
        y_1, y_2 = torch.chunk(y, 2, dim=1)
        #y_out = torch.cat([self.norm(y_1), y_2], dim=1)
        y_out = torch.cat([self.norm(y_1), self.identity(y_2)], dim=1)
        out1 = self.Con1X1(y_out)
        out2 = self.Con3X3(y_out)
        out3 = self.Con5X5(y_out)
        OUT = torch.cat([out1, out2, out3], 1)
        # print("输出===",X_1.shape,OUT.shape)

        return X_1, OUT




