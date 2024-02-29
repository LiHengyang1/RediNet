import numpy as np
import os
import glob
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datetime import datetime
from torch import optim
import scipy.io as scio
import h5py
# from torchsummary import summary
##########################################################################################################
PPath = r"D:\LHY\NN3D04"   # This path needs to be the absolute path to the folder where the dataset is stored
# ////////////////////////////////////////////////////////////////////////////////////////
class TRAIN_Loader(Dataset):

    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.image_path = glob.glob(os.path.join(data_path, 'TRAINimage0/*.mat'))

    def __getitem__(self, index):
        # 读取训练图片和标签图片
        image_path = self.image_path[index]
        image = scio.loadmat(image_path)
        image = image['t']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        label_path = image_path.replace('TRAINimage0', 'TRAINlabel0')
        label = scio.loadmat(label_path)
        label = label['t']
        label = label.reshape(1, label.shape[0], label.shape[1], label.shape[2])
        label = label / 255
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.image_path)


class VALI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.image_path = glob.glob(os.path.join(data_path, 'VALIimage0/*.mat'))

    def __getitem__(self, index):
        image_path = self.image_path[index]
        image = scio.loadmat(image_path)
        image = image['t']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        label_path = image_path.replace('VALIimage0', 'VALIlabel0')
        label = scio.loadmat(label_path)
        label = label['t']
        label = label.reshape(1, label.shape[0], label.shape[1], label.shape[2])
        label = label / 255
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.image_path)


TRAINset = TRAIN_Loader(PPath)
print("数据个数：", len(TRAINset))
VALIset = VALI_Loader(PPath)
print("数据个数：", len(VALIset))

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
        #         x4 = self.down3(x3)
        #
        #         x = self.up3(x4, x3)
        x = self.up2(x3, x2)
        x = self.up1(x, x1)

        x = self.up01(x)
        x = self.up02(x)
        x = self.up03(x)

        x = self.outConv(x)

        return x

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class My_loss_abs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow(x - y, 2))


class My_loss_phase1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(1 - torch.cos(2 * math.pi * (x - y)))

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def train_net(net, device, train_loader, vali_loader, criterion, optimizer, epoch, batch_size, best_loss):
    # 加载训练集
    net.train()
    y0 = 0
    # 按照batch_size开始训练

    for image, label in train_loader:
        optimizer.zero_grad()
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        pred = net(image)
        loss = criterion(pred, label)
        if (y0 % 10) == 0:
            print('Loss/train', loss.item())
        y0 += 1
        loss.backward()
        optimizer.step()
    net.eval()
    loss_WholeEpoch = 0
    with torch.no_grad():
        for image, label in vali_loader:
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(image)
            loss = criterion(pred, label)
            loss_WholeEpoch += loss
        loss_WholeEpoch = loss_WholeEpoch * batch_size / len(VALIdataset)
        print('Loss/VALI %f' % loss_WholeEpoch)

    if loss_WholeEpoch < best_loss:
        best_loss = loss_WholeEpoch
        torch.save(net.state_dict(), 'md17.pth')
        print('model saved ')
    return best_loss


########################################################################################################################
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet3D(1)
    net.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(net.parameters(), lr=30e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 6, eta_min=2e-6, last_epoch=-1)
    # optimizer = torch.optim.Adam(net.parameters(), lr=2e-6)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 6, eta_min=2e-7, last_epoch=-1)

    # criterion = My_loss_phase1()
    criterion = My_loss_abs()

    TRAINdata_path = PPath
    VALIdata_path = PPath
    batch_size = 64
    TRAINdataset = TRAIN_Loader(TRAINdata_path)
    train_loader = torch.utils.data.DataLoader(dataset=TRAINdataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    VALIdataset = VALI_Loader(VALIdata_path)
    vali_loader = torch.utils.data.DataLoader(dataset=VALIdataset,
                                              batch_size=batch_size,
                                              shuffle=True)


    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv3d:
            nn.init.xavier_uniform_(m.weight)


    net.apply(init_weights)
    #
    # net.load_state_dict(torch.load('md17.pth', map_location=device))

    print('training on', device)

    epochs = 1000

    best_loss = float('0.5')
    for epoch in range(epochs):
        tic = datetime.now()
        best_loss = train_net(net, device, train_loader, vali_loader, criterion, optimizer, epoch, batch_size,
                              best_loss)
        toc = datetime.now()
        print('EPOCH %f consumed: %f seconds , LR = %f' % (
            epoch + 1, (toc - tic).total_seconds(), optimizer.state_dict()['param_groups'][0]['lr']))
        scheduler.step()

