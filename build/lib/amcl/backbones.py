from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18, resnet50


def resnet_cifar(depth: int = 18):
    """ResNet with the CIFAR/STL stem used by the original small-image code."""
    model = resnet18(weights=None) if depth == 18 else resnet50(weights=None) if depth == 50 else None
    if model is None:
        raise ValueError("depth must be 18 or 50")
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, dim
