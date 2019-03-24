import torch


def size(net):
    pp = 0
    for p in list(net.parameters()):
        nn = 1
        for s in list(p.size()): 
            nn = nn * s
        pp += nn 
    return pp


def inspect_model(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images = torch.randn(4, 3, 64, 64).to(device)

    net = model(num_classes=2).to(device)

    logits = net(images)
    if images.shape[-2:] != logits.shape[-2:]:
        raise ValueError(f'Output sized {logits.shape[-2:]} while {images.shape[-2:]} expected')

    print(size(net), model.__name__, sep='\t')
