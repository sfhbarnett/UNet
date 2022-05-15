import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inconv1 = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512,1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inconv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class down(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(down, self).__init__()
        self.level = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.level(x)
        return x


class inconv(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(inconv, self).__init__()
        self.step = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.step(x)
        return x


class Up(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(Up, self).__init__()
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.ConvTranspose2d(inchannels, inchannels//2, stride=2, kernel_size=2)
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(outconv, self).__init__()
        self.step = nn.Conv2d(inchannels, outchannels, kernel_size=1)

    def forward(self, x):
        x = self.step(x)
        return x