"""
Made by @nizhib
"""

import torch
from torchvision import transforms

from models import unet_resnext50


def sanitize(state_dict):
    cpu = torch.device('cpu')
    sanitized = dict()
    for key in state_dict:
        if key.startswith('module.'):
            sanitized[key[7:]] = state_dict[key].to(cpu)
        else:
            sanitized[key] = state_dict[key].to(cpu)
    return sanitized


def load_state(path):
    state = torch.load(path, map_location='cpu')
    if 'state_dict' in state:
        state = state['state_dict']
    state = sanitize(state)
    return state


class Segmentator(object):
    size = (320, 240)

    meanstd = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    normalize = transforms.Normalize(**meanstd)
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.Pad((8, 0), padding_mode='reflect'),
        transforms.ToTensor(),
        normalize
    ])

    def __init__(self):
        self.net = unet_resnext50(num_classes=1)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load(self, checkpoint_path):
        state = load_state(checkpoint_path)
        self.net.load_state_dict(state)
        self.net.eval()
        self.net.to(self.device)

    @torch.no_grad()
    def predict(self, image):
        image = self.preprocess(image)
        tensor = torch.stack((image,)).to(self.device)
        logits = self.net(tensor)
        probs = torch.sigmoid(logits).data[0, 0, :, 8:-8].to('cpu').numpy()
        return probs
