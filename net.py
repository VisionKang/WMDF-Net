import torch

import sys

import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_simple import *
from mamba_module import *
from Method import *
from MambaHSI import *
from DWT_tool import *
from model.refine import Refine
from torchvision import transforms
from thop import profile, clever_format


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        base_filter = 16
        self.base_filter = base_filter
        self.stride = 1
        self.patch_size = 1
        self.deep_fusion1 = CrossMamba(base_filter)
        self.deep_fusion2 = CrossMamba(base_filter)
        self.ms_to_token = PatchEmbed(in_chans=base_filter,embed_dim=16,patch_size=self.patch_size,stride=self.stride)
        self.pan_to_token = PatchEmbed(in_chans=base_filter,embed_dim=16,patch_size=self.patch_size,stride=self.stride)


        self.wavt = DWT(32)

        self.mamba_spa_e = MambaHSI()
        self.pan_encoder = nn.Sequential(nn.Conv2d(1,base_filter,3,1,1),
                                         HinResBlock(base_filter,base_filter),
                                         HinResBlock(base_filter,base_filter),
                                         HinResBlock(base_filter,base_filter))


        self.ms_encoder = nn.Sequential(nn.Conv2d(4,base_filter,3,1,1),
                                        HinResBlock(base_filter,base_filter),
                                        HinResBlock(base_filter,base_filter),
                                        HinResBlock(base_filter,base_filter))
        self.yu_fuse = nn.Sequential(
            nn.Conv2d(32, 16, 1, 1, 0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 1, 1,0)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(16, 32, 1, 1, 0)
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(16, 32, 1, 1, 0)
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU()
        )


        self.recon_layer = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3, 1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 4, 1, 1, padding=0)
        )
        self.patchunembe = PatchUnEmbed(base_filter)
        self.output = Refine(48, 4)

    def forward(self, ms, pan):
        pt = pan.float().cuda()  # (1,1,128,128)
        mt = ms.float().cuda()  # (1,4,32,32)
        # mt_4 = upsample(ms,128,128)  # (1,4,128,128)
        mt_4 = F.interpolate(mt, (128, 128), mode="bicubic", align_corners=True)
        pt_4 = F.interpolate(pt, (32, 32), mode="bicubic", align_corners=True)
        # print("pt_4_shape={},mt_4shape={}".format(pt_4.shape,mt_4.shape))
        # ————浅层特征提取———— #
        mt32 = self.ms_encoder(mt)  # (1,16,32,32)
        pt_4_32 = self.pan_encoder(pt_4)  # (1,16,32,32)
        cat_32 = torch.cat([mt32, pt_4_32], 1)  # (1,32,32,32)

        mt128 = self.ms_encoder(mt_4)  # (1,16,128,128)
        pt128 = self.pan_encoder(pt)  # (1,16,128,128)
        cat_128 = torch.cat([mt128, pt128], 1)  # (1,32,128,128)
        # in_w1 = self.yu_fuse(cat_128)  # (1,32,128,128)
        in_w1 = cat_128  # (1,32,128,128)

        fH1, fL1 = self.wavt(in_w1)  # torch.Size([1, 16, 64, 64])  torch.Size([1, 32, 64, 64])
        fH2, fL2 = self.wavt(fL1)  # torch.Size([1, 16, 32, 32])  torch.Size([1, 32, 32, 32])

        in_mamba1 = cat_32  # +fL2   (1,32,32,32)
        out_mamba1_spa_x, out_mamba1_spe_x= self.mamba_spa_e(in_mamba1)  # (1,16,32,32)
        # print("11=out_mamba1_spa_x, out_mamba1_spe_x",out_mamba1_spa_x.shape, out_mamba1_spe_x.shape)
        # out_mamba1_spa_x, out_mamba1_spe_x == torch.Size([4, 16, 32, 32]) torch.Size([4, 16, 32, 32])
        out_mamba1_spa_x = self.ms_to_token(out_mamba1_spa_x)
        out_mamba1_spe_x = self.ms_to_token(out_mamba1_spe_x)
        # print("22=out_mamba1_spa_x, out_mamba1_spe_x", out_mamba1_spa_x.shape, out_mamba1_spe_x.shape)

        residual_ms_f = 0
        ms_f, residual_ms_f = self.deep_fusion1(out_mamba1_spe_x, residual_ms_f, out_mamba1_spa_x,32,32)
        ms_f, residual_ms_f = self.deep_fusion2(ms_f, residual_ms_f, out_mamba1_spa_x,32,32)
        out_mamba1 = self.patchunembe(ms_f, (32, 32))
        # print("out_mamba1", out_mamba1.shape)

        layer1_in = out_mamba1+fH2  # (1,16,32,32)
        layer1_out = self.layer1(layer1_in)  # (1,32,64,64)

        in_mamba2 = layer1_out + fL1  # (1,32,64,64)
        out_mamba2_spa_x, out_mamba2_spe_x = self.mamba_spa_e(in_mamba2)  # (1,16,64,64)
        # print("33=out_mamba2_spa_x, out_mamba2_spe_x", out_mamba2_spa_x.shape, out_mamba2_spe_x.shape)
        out_mamba2_spa_x = self.ms_to_token(out_mamba2_spa_x)
        out_mamba2_spe_x = self.ms_to_token(out_mamba2_spe_x)
        # print("44=out_mamba2_spa_x, out_mamba2_spe_x", out_mamba2_spa_x.shape, out_mamba2_spe_x.shape)
        residual_ms_f = 0
        ms_f, residual_ms_f = self.deep_fusion1(out_mamba2_spe_x, residual_ms_f, out_mamba2_spa_x,64,64)
        ms_f, residual_ms_f = self.deep_fusion2(ms_f, residual_ms_f, out_mamba2_spa_x,64,64)
        out_mamba2 = self.patchunembe(ms_f, (64, 64))
        # print("out_mamba2",out_mamba2.shape)
        # output is out_mamba2 torch.Size([4, 16, 64, 64]) yes

        layer2_in = out_mamba2 + fH1  # (1,16,64,64)
        layer2_out = self.layer2(layer2_in)  # (1,32,128,128)

        in_mamba3 = layer2_out # (1,32,128,128)

        out_mamba3_spa_x, out_mamba3_spe_x = self.mamba_spa_e(in_mamba3)  # (1,16,64,64)
        # print("55=out_mamba3_spa_x, out_mamba3_spe_x", out_mamba3_spa_x.shape, out_mamba3_spe_x.shape)
        out_mamba3_spa_x = self.ms_to_token(out_mamba3_spa_x)
        out_mamba3_spe_x = self.ms_to_token(out_mamba3_spe_x)
        # print("66=out_mamba3_spa_x, out_mamba3_spe_x", out_mamba3_spa_x.shape, out_mamba3_spe_x.shape)
        residual_ms_f = 0
        ms_f, residual_ms_f = self.deep_fusion1(out_mamba3_spe_x, residual_ms_f, out_mamba3_spa_x, 128, 128)
        ms_f, residual_ms_f = self.deep_fusion2(ms_f, residual_ms_f, out_mamba3_spa_x, 128, 128)
        out_mamba3 = self.patchunembe(ms_f, (128, 128))





        # out_mamba3 = self.mamba_spa_e(in_mamba3)  # (1,16,128,128)

        out = torch.cat([out_mamba3, in_w1], 1)  # (1,48,128,128)

        # hrms = self.recon_layer(out) + mt_4
        hrms = self.output(out) + mt_4


        # print("hrms={}".format(hrms.shape))

        return hrms



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(4, 4, 32, 32).to(device)
    z = torch.randn(4, 1, 128, 128).to(device)

    Net = Generator().to(device)
    out = Net(x, z)
    print(out.shape)
    flops, params = profile(Net, inputs=(x, z), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print(f"Total FLOPs: {flops}")
    print(f"Total Params: {params}")




