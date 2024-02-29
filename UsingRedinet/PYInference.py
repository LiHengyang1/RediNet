import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime




# ////////////////////////////////////////////////////////////////////////////////////////

class DownResblock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, use_1x1conv=True, strides=1):
        super(DownResblock, self).__init__()
        self.downsamp = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.downsamp(x)
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        Y += x
        return F.relu(Y)


class UpResblock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, use_1x1conv=True, strides=1, bilinear=False):
        super(UpResblock, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        Y = F.relu(self.bn1(self.conv1(x1)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x1 = self.conv3(x1)
        Y += x1
        return F.relu(Y)

class ResidualClassic(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_1x1conv=True, strides=1):
        super(ResidualClassic, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        Y += x
        return F.relu(Y)


class UpResSupp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, use_1x1conv=True, strides=1, bilinear=False):
        super(UpResSupp, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.up(x)
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        Y += x
        return F.relu(Y)
################################################################################################################

class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes=2, batch_normal=False, bilinear=True):
        super(UNet3D, self).__init__()
        nLayer = [1, 128, 256, 512]
        nLayer0 = [1, 128, 64, 32, 16]

        self.inConv = ResidualClassic(1, nLayer[1], use_1x1conv=True, strides=1)
        self.down1 = DownResblock(nLayer[1], nLayer[2], use_1x1conv=True, strides=1)
        self.down2 = DownResblock(nLayer[2], nLayer[3], use_1x1conv=True, strides=1)
        #         self.down3 = DownResblock(nLayer[3], nLayer[4], use_1x1conv=True, strides=1)

        #         self.up3 = UpResblock(nLayer[4], nLayer[3], use_1x1conv=True, strides=1, bilinear=False)
        self.up2 = UpResblock(nLayer[3], nLayer[2], use_1x1conv=True, strides=1, bilinear=False)
        self.up1 = UpResblock(nLayer[2], nLayer[1], use_1x1conv=True, strides=1, bilinear=False)

        self.up01 = UpResSupp(nLayer0[1], nLayer0[2], use_1x1conv=True, strides=1, bilinear=False)
        self.up02 = UpResSupp(nLayer0[2], nLayer0[3], use_1x1conv=True, strides=1, bilinear=False)
        self.up03 = UpResSupp(nLayer0[3], nLayer0[4], use_1x1conv=True, strides=1, bilinear=False)

        self.outConv = ResidualClassic(nLayer0[4], 1, use_1x1conv=True, strides=1)

    def forward(self, x):
        x1 = self.inConv(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up2(x3, x2)
        x = self.up1(x, x1)

        x = self.up01(x)
        x = self.up02(x)
        x = self.up03(x)

        x = self.outConv(x)

        return x
#########################################################################################

def pyfun0(image):
    netMAT = UNet3D(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netMAT.to(device)
    netMAT.load_state_dict(torch.load('uc17.pth', map_location=device))
    netMAT.eval()
    image = image.reshape(1, 1, image.shape[0], image.shape[1], image.shape[2])

    img_tensor = torch.from_numpy(image)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    tic = datetime.now()
    pred = netMAT(img_tensor)
    toc = datetime.now()
    pred = np.array(pred.data.cpu()[0])[0]
    print(np.array((toc - tic).total_seconds()))
    return pred



###################################################################################################################