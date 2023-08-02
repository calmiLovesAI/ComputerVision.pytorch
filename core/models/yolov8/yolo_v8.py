import contextlib
from typing import List

import ast
import torch
import torch.nn as nn

from core.models.yolov8.modules import (C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                        Classify, Concat, Conv, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                        Ensemble, Focus, GhostBottleneck, GhostConv, Pose, Segment)
from core.models.yolov8.torch_utils import initialize_weights
from core.utils.show import colorstr
from core.utils.ultralytics_ops import make_divisible


class Yolo8(nn.Module):
    def __init__(self, scale: List, num_classes=80, ch=3):
        """
        :param scale: [depth, width, max_channels] 不同尺寸的yolo_v8模型的参数
        :param num_classes: 类别数，默认数据集为COCO，80类
        :param ch: 输入通道数，默认为3
        """
        super().__init__()
        self.depth, self.width, self.max_channels = scale
        self.num_classes = num_classes
        layers = [
            # backbone
            Conv(c1=ch, c2=self._ac(64), k=3, s=2),  # 0
            Conv(c1=self._ac(64), c2=self._ac(128), k=3, s=2),  # 1
            C2f(c1=self._ac(128), c2=self._ac(128), n=self._get_n(3), shortcut=True),
            Conv(c1=self._ac(128), c2=self._ac(256), k=3, s=2),  # 3
            C2f(c1=self._ac(256), c2=self._ac(256), n=self._get_n(6), shortcut=True),
            Conv(c1=self._ac(256), c2=self._ac(512), k=3, s=2),  # 5
            C2f(c1=self._ac(512), c2=self._ac(512), n=self._get_n(6), shortcut=True),
            Conv(c1=self._ac(512), c2=self._ac(1024), k=3, s=2),  # 7
            C2f(c1=self._ac(1024), c2=self._ac(1024), n=self._get_n(3), shortcut=True),
            SPPF(c1=self._ac(1024), c2=self._ac(1024), k=5),  # 9
            # head
            nn.Upsample(scale_factor=2), Concat(dimension=1),
            C2f(c1=(self._ac(1024) + self._ac(512)), c2=self._ac(512), n=self._get_n(3)),  # 12
            nn.Upsample(scale_factor=2), Concat(dimension=1),
            C2f(c1=(self._ac(512) + self._ac(256)), c2=self._ac(256), n=self._get_n(3)),  # 15
            Conv(c1=self._ac(256), c2=self._ac(256), k=3, s=2), Concat(dimension=1),
            C2f(c1=(self._ac(256) + self._ac(512)), c2=self._ac(512), n=self._get_n(3)),  # 18
            Conv(c1=self._ac(512), c2=self._ac(512), k=3, s=2), Concat(dimension=1),
            C2f(c1=(self._ac(512) + self._ac(1024)), c2=self._ac(1024), n=self._get_n(3)),  # 21
            Detect(nc=self.num_classes, ch=(self._ac(256), self._ac(512), self._ac(1024)))
        ]

        self.model = nn.Sequential(*layers)  # 总共23层

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose)):
            s = 256  # 2x min stride
            m.inplace = True
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once

        initialize_weights(self)

    def _get_n(self, n):
        return max(round(n * self.depth), 1) if n > 1 else n

    def _ac(self, c):
        """
        Adjust channel
        :param c: channel
        :return: channel after adjusting
        """
        if c != self.num_classes:
            return make_divisible(min(c, self.max_channels) * self.width, 8)
        else:
            return c

    def forward(self, x):
        """
        Perform a forward pass through the network.
        :param x:  The input tensor to the model
        :return:
        如果YOLOv8在train模式下，那么输出是list of torch.Tensor,
        [torch.Size([bs, nc + 16*4, 80, 80]), torch.Size([bs, nc + 16*4, 40, 40]), torch.Size([bs, nc + 16*4, 20, 20])]
        如果YOLOv8在eval模式下，输出是一个元组，包含两部分，第一部分是一个Tensor, shape是[bs, nc + 4, 8400(80*80+40*40+20*20)]
        第二部分就是train模式下的输出
        """
        saved_layer_outputs = {}
        for i, m in enumerate(self.model):
            # Concat层，输入为上一层的输出和saved_layer_outputs中某一层的输出
            if i == 11:
                x = m([x, saved_layer_outputs["layer-6"]])
            elif i == 14:
                x = m([x, saved_layer_outputs["layer-4"]])
            elif i == 17:
                x = m([x, saved_layer_outputs["layer-12"]])
            elif i == 20:
                x = m([x, saved_layer_outputs["layer-9"]])
            elif i == 22:
                # 最后一层，输入为第15、18、21层的输出
                x = m([saved_layer_outputs["layer-15"], saved_layer_outputs["layer-18"], x])
            else:
                # 其它层，输入为上一层的输出
                x = m(x)
            if i == 4 or i == 6 or i == 9 or i == 12 or i == 15 or i == 18 or i == 21:
                saved_layer_outputs[f"layer-{i}"] = x
        return x


def get_yolo8_n(nc=80):
    # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
    return Yolo8(scale=[0.33, 0.25, 1024], num_classes=nc)


def get_yolo8_s(nc=80):
    # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
    return Yolo8(scale=[0.33, 0.50, 1024], num_classes=nc)


def get_yolo8_m(nc=80):
    # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
    return Yolo8(scale=[0.67, 0.75, 768], num_classes=nc)


def get_yolo8_l(nc=80):
    # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
    return Yolo8(scale=[1.00, 1.00, 512], num_classes=nc)


def get_yolo8_x(nc=80):
    # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
    return Yolo8(scale=[1.00, 1.25, 512], num_classes=nc)
