"""
Made by @nizhib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import conv3x3


class DecoderBlock(nn.Module):
    def __init__(self, down_channels, left_channels):
        super().__init__()

        self.upconv = conv3x3(down_channels, left_channels)
        self.conv1 = conv3x3(left_channels * 2, left_channels)
        self.bn1 = nn.BatchNorm2d(left_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(left_channels, left_channels)
        self.bn2 = nn.BatchNorm2d(left_channels)
        self.relu2 = nn.ReLU()

    def forward(self, down, left):
        if down.shape[2] != left.shape[2]:
            size = (left.shape[2], left.shape[3])
            down = F.interpolate(down, size=size, mode='bilinear', align_corners=False)
        x = self.upconv(down)
        x = torch.cat((left, x), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.blocks = nn.ModuleList()
        for down_channels, left_channels in zip(features[-1:0:-1], features[-2::-1]):
            self.blocks.append(DecoderBlock(down_channels, left_channels))

    def forward(self, acts):
        up = acts[-1]
        for left, block in zip(acts[-2::-1], self.blocks):
            up = block(up, left)
        return up
