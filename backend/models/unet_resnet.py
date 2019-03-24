"""
Made by @nizhib
"""

from torch import nn

from .classification import resnext50, resnext101
from .decoders import UNetDecoder
from .encoders import ResNetEncoder
from .layers import conv3x3, Upsample
from .utils import inspect_model

__all__ = ['unet_resnext50', 'unet_resnext101']


class UResNet(nn.Module):
    def __init__(self, arch, num_classes, pretrained=False):
        super().__init__()

        self.encoder = ResNetEncoder(arch, pretrained=pretrained)
        self.decoder = nn.Sequential(
            UNetDecoder(self.encoder.features),
            Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            conv3x3(self.encoder.features[0], self.encoder.features[0])
        )
        self.final = nn.Conv2d(self.encoder.features[0], num_classes, 1)

    def forward(self, x):
        acts = self.encoder(x)
        x = self.decoder(acts)
        x = self.final(x)
        return x


def unet_resnext50(num_classes=1, pretrained=False):
    return UResNet(resnext50, num_classes=num_classes, pretrained=pretrained)


def unet_resnext101(num_classes=1, pretrained=False):
    return UResNet(resnext101, num_classes=num_classes, pretrained=pretrained)


if __name__ == '__main__':
    for exported in __all__:
        inspect_model(globals()[exported])
