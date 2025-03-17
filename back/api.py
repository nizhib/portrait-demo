"""
Made by @nizhib
"""

from typing import ClassVar

import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as F

from models import unet_resnext50


class Segmenter:
    size: ClassVar[tuple[int, int]] = (320, 320)
    mean: ClassVar[list[float]] = [0.485, 0.456, 0.406]
    std: ClassVar[list[float]] = [0.229, 0.224, 0.225]

    def __init__(self, step: int = 32) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = unet_resnext50(num_classes=1, pretrained=True)
        self.net.eval()
        self.net.to(self.device)

        self.step = step
        self.pt = 0
        self.pr = 0
        self.pb = 0
        self.pl = 0

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image.thumbnail(self.size)

        tensor = F.to_tensor(image)

        h, w = tensor.shape[-2:]
        self.pl = (w % self.step + 1) // 2
        self.pt = (h % self.step) // 2
        self.pr = (w % self.step) // 2
        self.pb = (h % self.step + 1) // 2
        tensor = F.pad(
            tensor, [self.pl, self.pt, self.pr, self.pb], padding_mode="reflect"
        )

        tensor = F.normalize(tensor, self.mean, self.std)
        tensor = tensor.to(self.device)

        batch = torch.stack([tensor])

        return batch

    def postprocess(self, logits: torch.Tensor) -> np.ndarray:
        logits = torch.squeeze(logits)

        h, w = logits.shape
        logits = logits[self.pt : h - self.pb, self.pl : w - self.pr]

        probs = torch.sigmoid(logits)
        probs = probs.to("cpu")
        probs = probs.numpy()

        return probs

    @torch.no_grad()
    def predict(self, image: Image.Image) -> np.ndarray:
        batch = self.preprocess(image)
        logits = self.net(batch)
        probs = self.postprocess(logits)
        return probs
