import torch
import torch.nn as nn
import torch.nn.functional as F


class Fusionloss2(nn.Module):
    def __init__(self, weight_sp=1000, weight_fp=1000):
        super(Fusionloss2, self).__init__()
        self.weight_sp = weight_sp
        self.weight_fp = weight_fp
    
    def forward(self,  image_ir, fusion):

        loss_in = F.l1_loss(image_ir, fusion)

        fft_ir = torch.fft.fft2(image_ir)
        fft_fusion = torch.fft.fft2(fusion)
        # 振幅损失
        amplitude_ir = torch.abs(fft_ir)
        amplitude_fusion = torch.abs(fft_fusion)
        loss_amplitude = F.l1_loss(amplitude_ir, amplitude_fusion)
        # 相位损失
        phase_ir = torch.angle(fft_ir)
        phase_fusion = torch.angle(fft_fusion)
        loss_phase = F.l1_loss(phase_ir, phase_fusion)
        # 总损失
        loss_total = (self.weight_sp * loss_in +(loss_amplitude + 0 * loss_phase) * self.weight_fp)

        print("loss_in = {:.5f} , loss_amplitude = {:.5f} , loss_phase = {:.5f}, loss_total = {:.5f}".format(
            loss_in * 1000, loss_amplitude * 1000, loss_phase * 1000, loss_total))
        
        return loss_total


class Fusionloss11(nn.Module):
    def __init__(self, weight_sp=1000, weight_grag=1000, weight_fp=1000, weight_pha=1000):
        super(Fusionloss11, self).__init__()
        self.weight_sp = weight_sp
        self.weight_grag = weight_grag
        self.weight_fp = weight_fp
        self.weight_pha = weight_pha
        self.sobelconv = Sobelxy()
        # self.sobelconv_ms = Sobelxy_ms()

    def forward(self, image_pan, image_ir, fusion, fusion_ap):
        loss_in = F.l1_loss(image_ir, fusion)

        fft_ir = torch.fft.fft2(image_ir)
        fft_fusion = torch.fft.fft2(fusion)
        # 振幅损失
        amplitude_ir = torch.abs(fft_ir)
        amplitude_fusion = torch.abs(fft_fusion)
        loss_amplitude = F.l1_loss(amplitude_ir, amplitude_fusion)
        # 相位损失
        phase_ir = torch.angle(fft_ir)
        phase_fusion = torch.angle(fft_fusion)
        loss_phase = F.l1_loss(phase_ir, phase_fusion)
        # 梯度损失
        pan_grad = self.sobelconv(image_pan)
        generate_img_grad = self.sobelconv(fusion_ap)
        loss_grad = F.l1_loss(generate_img_grad, pan_grad)
        # 总损失
        loss_total = (self.weight_sp * loss_in + self.weight_grag * loss_grad + loss_amplitude * self.weight_fp +
                      self.weight_pha * loss_phase)

        print("loss_in = {:.5f} , loss_grad = {:.5f} , loss_amplitude = {:.5f} , loss_phase = {:.5f}, loss_total = {:.5f}".format(
            loss_in * 1000, loss_grad*1000, loss_amplitude * 1000, loss_phase * 1000, loss_total))

        return loss_total



class Fusionloss1(nn.Module):
    def __init__(self):
        super(Fusionloss1, self).__init__()
        self.sobelconv = Sobelxy()
        self.sobelconv_ms = Sobelxy_ms()
    def forward(self, image_vis, image_ir, fusion, fusion_ap):
        # x_in_max = torch.max(image_vis, image_ir)
        # loss_in = F.mse_loss(x_in_max, generate_img)
        #print("使用li损失",image_ir.shape, image_vis.shape, fusion.shape, fusion_ap.shape)
        loss_in = F.l1_loss(fusion, image_ir)
        vis_grad = self.sobelconv(image_vis)
        # ir_grad = self.sobelconv_ms(image_ir)
        generate_img_grad = self.sobelconv(fusion_ap)
        # x_grad_joint = torch.max(vis_grad, ir_grad)
        loss_grad = F.l1_loss(generate_img_grad, vis_grad)

        loss_total = loss_in*10
        return loss_total

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

class Sobelxy_ms(nn.Module):
    def __init__(self):
        super(Sobelxy_ms, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx = F.conv2d(x, self.weightx.repeat(1, 4, 1, 1), padding=1)
        sobely = F.conv2d(x, self.weighty.repeat(1, 4, 1, 1), padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)
