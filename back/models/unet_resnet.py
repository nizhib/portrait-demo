"""
Made by @nizhib
"""

from typing import Callable

import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, conv3x3, resnext50_32x4d

from .decoders import UNetDecoder
from .encoders import ResNetEncoder

__all__ = ["unet_resnext50"]

model_urls = {
    "unet_resnext50": "https://github.com/nizhib/portrait-demo/releases/download/v0.3/unet_resnext50-46a04131.pth"
}


class UResNet(nn.Module):
    def __init__(self, arch: Callable[..., ResNet], num_classes: int) -> None:
        super().__init__()

        self.encoder = ResNetEncoder(arch)
        self.decoder = nn.Sequential(
            UNetDecoder(self.encoder.features),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv3x3(self.encoder.features[0], self.encoder.features[0]),
        )
        self.final = nn.Conv2d(self.encoder.features[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts = self.encoder(x)
        x = self.decoder(acts)
        x = self.final(x)
        return x


def unet_resnext50(
    num_classes: int = 1, pretrained: bool = False, progress: bool = True
) -> UResNet:
    model = UResNet(resnext50_32x4d, num_classes=num_classes)
    if pretrained:
        if num_classes != 1:
            raise ValueError("Pretrained weights are for num_classes=1 only")
        state_dict = load_state_dict_from_url(
            model_urls["unet_resnext50"], progress=progress
        )
        mkeys = set(model.state_dict().keys())
        skeys = set(state_dict.keys())
        print(list(mkeys - skeys))
        print(list(skeys - mkeys))
        model.load_state_dict(state_dict)
    return model
