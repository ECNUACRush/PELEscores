""" Full assembly of the parts to form the complete network """
import torch

"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F
from torch.autograd import Variable
from .unet_parts import *



class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, m):
        x, y = m
        return torch.cat([x, y], 1)


def concatenate_layer(in_channel1, in_channel2, out_channel3):
    return nn.Sequential(*[
        Concat(),
        nn.Conv2d(in_channel1 + in_channel2, out_channel3, (1, 1), (1, 1), (0, 0), bias=False),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(out_channel3),
    ])


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        base_channel = 80
        expand_ratio = 2
        self.inc = DoubleConv(n_channels, base_channel)
        self.down1 = PDown(base_channel, base_channel * expand_ratio)
        self.down2 = PDown(base_channel * expand_ratio, base_channel * (expand_ratio ** 2))
        self.down3 = PDown(base_channel * (expand_ratio ** 2), base_channel * (expand_ratio ** 3))
        self.down4 = PDown(base_channel * (expand_ratio ** 3), base_channel * (expand_ratio ** 3))
        self.up1 = Up(base_channel * (expand_ratio ** 3 + expand_ratio ** 3), base_channel * (expand_ratio ** 2),
                      bilinear)
        self.up2 = Up(base_channel * (expand_ratio ** 2 + expand_ratio ** 2), base_channel * expand_ratio, bilinear)
        self.up3 = Up(base_channel * (expand_ratio + expand_ratio), base_channel, bilinear)
        self.up4 = Up(base_channel * (2), base_channel, bilinear)
        self.outc = OutConv(base_channel, n_classes)

        self.s_up1 = Up(base_channel * (expand_ratio ** 3 + expand_ratio ** 3), base_channel * (expand_ratio ** 2),
                        bilinear)
        self.s_up2 = Up(base_channel * (expand_ratio ** 2 + expand_ratio ** 2), base_channel * expand_ratio, bilinear)
        self.s_up3 = Up(base_channel * (expand_ratio + expand_ratio), base_channel, bilinear)
        self.s_up4 = Up(base_channel * (2), base_channel, bilinear)
        self.s_outc = OutConv(base_channel, n_classes)

        self.s_d1 = concatenate_layer(base_channel, base_channel, base_channel)
        self.s_d2 = concatenate_layer(base_channel * expand_ratio, base_channel, base_channel * expand_ratio)
        self.s_d3 = concatenate_layer(base_channel * (expand_ratio ** 2), base_channel * expand_ratio,
                                      base_channel * (expand_ratio ** 2))
        self.s_d4 = concatenate_layer(base_channel * (expand_ratio ** 3), base_channel * (expand_ratio ** 2),
                                      base_channel * (expand_ratio ** 3))
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)  # x2
        x3 = self.down2(x2)  # x4
        x4 = self.down3(x3)  # x8
        x5 = self.down4(x4)  # x16
        y4 = self.up1(x5, x4)
        y3 = self.up2(y4, x3)
        y2 = self.up3(y3, x2)
        y1 = self.up4(y2, x1)
        # torch.Size([16, 64, 384, 384]) torch.Size([16, 128, 192, 192]) torch.Size([16, 256, 96, 96]) torch.Size([16, 512, 48, 48]) torch.Size([16, 512, 24, 24]) torch.Size([16, 256, 48, 48]) torch.Size([16, 128, 96, 96]) torch.Size([16, 64, 192, 192]) torch.Size([16, 64, 384, 384])
        mask = self.outc(y1)  # y1,y2,y3,y4
        new_x1 = self.s_d1([x1, y1])
        new_x2 = self.s_d2([x2, y2])
        new_x3 = self.s_d3([x3, y3])
        new_x4 = self.s_d4([x4, y4])
        y = self.s_up1(x5, new_x4)
        y = self.s_up2(y, new_x3)
        y = self.s_up3(y, new_x2)
        y = self.s_up4(y, new_x1)
        logit = self.s_outc(y)
        return torch.clip(mask,-5,5),torch.clip(logit,-50,50)


if __name__ == '__main__':
    net = UNet(n_channels=1, n_classes=1)
    print(net)

"""
import torch.nn.functional as F

from .unet_parts import *

class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()
    def forward(self,m):
        x,y = m
        return torch.cat([x,y],1)

def concatenate_layer(in_channel1,in_channel2,out_channel3):
    return nn.Sequential(*[
                Concat(),
                nn.Conv2d(in_channel1+in_channel2,out_channel3,(1,1),(1,1),(0,0),bias=False),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(out_channel3),
        ])

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.s_up1 = Up(1024, 256, bilinear)
        self.s_up2 = Up(512, 128, bilinear)
        self.s_up3 = Up(256, 64, bilinear)
        self.s_up4 = Up(128, 64, bilinear)
        self.s_outc = OutConv(64, n_classes)

        self.s_d1 = concatenate_layer(64,64,64)
        self.s_d2 = concatenate_layer(128,64,128)
        self.s_d3 = concatenate_layer(256,128,256)
        self.s_d4 = concatenate_layer(512,256,512)





    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1) # x2
        x3 = self.down2(x2) # x4
        x4 = self.down3(x3) # x8
        x5 = self.down4(x4) # x16
        y4 = self.up1(x5, x4)
        y3 = self.up2(y4, x3)
        y2 = self.up3(y3, x2)
        y1 = self.up4(y2, x1)
        # torch.Size([16, 64, 384, 384]) torch.Size([16, 128, 192, 192]) torch.Size([16, 256, 96, 96]) torch.Size([16, 512, 48, 48]) torch.Size([16, 512, 24, 24]) torch.Size([16, 256, 48, 48]) torch.Size([16, 128, 96, 96]) torch.Size([16, 64, 192, 192]) torch.Size([16, 64, 384, 384])
        mask = self.outc(y1) # y1,y2,y3,y4
        new_x1 = self.s_d1([x1,y1])
        new_x2 = self.s_d2([x2,y2])
        new_x3 = self.s_d3([x3,y3])
        new_x4 = self.s_d4([x4,y4])
        y = self.s_up1(x5,new_x4)
        y = self.s_up2(y,new_x3)
        y = self.s_up3(y,new_x2)
        y = self.s_up4(y,new_x1)
        logit = self.s_outc(y)
        return mask,logit

if __name__ == '__main__':
    net = UNet(n_channels=1, n_classes=1)
    print(net)
"""
