import numpy as np
import thop
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def initialize_weights(model):
    """Initialize model weights to random values."""
    print("初始化模型参数...")
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True