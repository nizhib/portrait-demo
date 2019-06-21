"""
Made by @nizhib
"""

import torch
import torch.nn as nn


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

    def __init__(self, arch, pretrained=False):
        super().__init__()

        backbone = arch(pretrained=pretrained)

        self.encoder0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )
        self.encoder1 = nn.Sequential(
            backbone.maxpool,
            backbone.layer1
        )
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.features = [
            module[-1].conv3.out_channels
            if 'conv3' in module[-1].__dict__['_modules']
            else module[-1].conv2.out_channels
            if 'conv2' in module[-1].__dict__['_modules']
            else module[0].out_channels
            for module in [
                self.encoder0,
                self.encoder1[-1],
                self.encoder2,
                self.encoder3,
                self.encoder4
            ]
        ]

    def forward(self, x):
        acts = []
        x = self.encoder0(x)
        # print(x.shape)
        acts.append(x)
        x = self.encoder1(x)
        # print(x.shape)
        acts.append(x)
        x = self.encoder2(x)
        # print(x.shape)
        acts.append(x)
        x = self.encoder3(x)
        # print(x.shape)
        acts.append(x)
        x = self.encoder4(x)
        # print(x.shape)
        acts.append(x)
        return acts


if __name__ == '__main__':
    from torchvision.models import resnet50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images = torch.randn(4, 3, 224, 224).to(device)

    encoder = ResNetEncoder(resnet50)
    print(encoder.features)

    activations = encoder(images)
    for activation in activations:
        print(activation.shape[1:])
