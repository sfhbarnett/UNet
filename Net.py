import torch.nn as nn
import torch.nn.functional as F
import torch


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inconv1 = inconv(n_channels,64)
        self.down1 = down(64,128)
        self.down2 = down(128,256)
        self.down3 = down(256,512)
        self.midconv = nn.Sequential(
            nn.Conv2d(512,1024,3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024,1024,3),
            nn.ReLU(inplace=True),
        )
        self.up1 = up2(1024,512)
        self.up2 = up2(512,256)
        self.up3 = up2(256,128)
        self.up4 = up2(128,64)
        self.outc = nn.Conv2d(64,n_classes,3)

    def forward(self, x):
        x1 = self.inconv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.midconv(x4)
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        print(x.shape)
        x = self.outc(x)
        return F.sigmoid(x)


class down(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(down,self).__init__()
        self.level = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(inchannels,outchannels,3),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels,outchannels,3),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.level(x)
        return x

class inconv(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(inconv,self).__init__()
        self.step = nn.Sequential(
            nn.Conv2d(inchannels,outchannels,3),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels,outchannels,3),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.step(x)
        return x


class up(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(up,self).__init__()
        self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels,outchannels,3),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels,outchannels,3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self,x1,x2):
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

class up2(nn.Module):
    def __init__(self,inchannels, outchannels):
        super(up2,self).__init__()
        self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.upconv = nn.Conv2d(inchannels,int(inchannels/2),3)
        self.crossconv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, 3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x2):
        x = self.up(x)
        x1 = self.upconv(x)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.crossconv(x)
        return x