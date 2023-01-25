"""
Made by @nizhib
"""

from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, WeightsEnum


class ResNetEncoder(nn.Module):
    """
    ResNetEncoder(resnet50) @ 224x224:
    [64, 256, 512, 1024, 2048]
    torch.Size([64, 112, 112])
    torch.Size([256, 56, 56])
    torch.Size([512, 28, 28])
    torch.Size([1024, 14, 14])
    torch.Size([2048, 7, 7])
    """

    def __init__(
        self, arch: Callable[..., ResNet], weights: Optional[WeightsEnum] = None
    ) -> None:
        super().__init__()

        backbone = arch(weights=weights)

        self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.encoder1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.features = [
            int(module[-1].conv3.out_channels)  # type: ignore
            if "conv3" in module[-1].__dict__["_modules"]
            else int(module[-1].conv2.out_channels)  # type: ignore
            if "conv2" in module[-1].__dict__["_modules"]
            else int(module[0].out_channels)  # type: ignore
            for module in [
                self.encoder0,
                self.encoder1[-1],
                self.encoder2,
                self.encoder3,
                self.encoder4,
            ]
        ]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        activations = []
        x = self.encoder0(x)
        activations.append(x)
        x = self.encoder1(x)
        activations.append(x)
        x = self.encoder2(x)
        activations.append(x)
        x = self.encoder3(x)
        activations.append(x)
        x = self.encoder4(x)
        activations.append(x)
        return activations
