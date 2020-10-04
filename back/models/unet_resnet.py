"""
Made by @nizhib
"""

from torch import nn
from torchvision.models.utils import load_state_dict_from_url

from .classification import resnext50
from .decoders import UNetDecoder
from .encoders import ResNetEncoder
from .layers import conv3x3, Upsample
from .utils import inspect_model

__all__ = ['unet_resnext50']

model_urls = {
    'unet_resnext50': 'https://sota.nizhib.ai/portrait/unet_resnext50-eae871c0.pth'
}


class UResNet(nn.Module):
    def __init__(self, arch, num_classes):
        super().__init__()

        self.encoder = ResNetEncoder(arch)
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


def unet_resnext50(num_classes=1, pretrained=False, progress=True):
    model = UResNet(resnext50, num_classes=num_classes)
    if pretrained:
        if num_classes != 1:
            raise ValueError('Pretrained weights are for num_classes=1 only')
        state_dict = load_state_dict_from_url(model_urls['unet_resnext50'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    for exported in __all__:
        inspect_model(globals()[exported])
