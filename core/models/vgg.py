import warnings

import torch
import torch.nn as nn

from core.utils.device import get_device
from core.utils.file_ops import load_state_dict_from_url

__all__ = [
    "BaseVGG",
    "get_vgg11",
    "get_vgg13",
    "get_vgg16",
    "get_vgg19",
]

VGG_STRUCTURES = {
    "11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
}

VGG_BN_WEIGHTS = {
    "11": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",

}


class BaseVGG(nn.Module):
    def __init__(self, vgg_type, c_in, use_bn=True, num_classes=1000):
        """
        :param vgg_type: str, vgg类型，可选择的有：11，13，16，19
        :param c_in: int, 输入通道数
        :param use_bn: bool, 是否使用BN层，默认为True
        :param num_classes: int, 最后一个Linear层输出通道数，默认为1000
        """
        super(BaseVGG, self).__init__()
        self.structure = VGG_STRUCTURES[vgg_type]

        self.features = self._make_layers(c_in, batch_norm=use_bn)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def _make_layers(self, c_in, batch_norm=False):
        layers = []
        in_channels = c_in
        for v in self.structure:
            if v == 'M':  # max pool layer
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:  # conv layer
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.classifier(x)
        return x


def get_vgg(vgg_type, end_layer=-1, pretrained=True, num_classes=1000, only_feature=True):
    """
    导出VGG模型
    :param vgg_type: str, vgg类型，可选择的有：11，13，16，19
    :param end_layer: int, feature结束层的序号，默认为-1，当only_feature为True时有效
    :param pretrained: bool，默认为True
    :param num_classes: int, 最后一个Linear层输出通道数，默认为1000
    :param only_feature: int, 仅返回特征提取层
    :return:
    """
    vgg = BaseVGG(vgg_type, 3, num_classes=num_classes)
    model_name = f"vgg{vgg_type}"
    n_layers = len(vgg.features)
    if end_layer < 0:
        end_layer = n_layers + end_layer
    if end_layer > n_layers:
        raise ValueError(
            f"The end_layer is {end_layer}, which is greater than the {model_name}'s total feature layers number: {n_layers}.")
    if pretrained:
        # 加载预训练模型
        state_dict = load_state_dict_from_url(url=VGG_BN_WEIGHTS["11"],
                                              model_dir="downloads/vgg11_ImageNet1K.pth",
                                              map_location=get_device())
        vgg.load_state_dict(state_dict)
    if only_feature and end_layer != n_layers:
        return vgg.features[:end_layer + 1]
    else:
        return vgg


def get_vgg11(end_layer=-1, pretrained=True, num_classes=1000, only_feature=True):
    return get_vgg("11", end_layer, pretrained, num_classes, only_feature)


def get_vgg13(end_layer=-1, pretrained=True, num_classes=1000, only_feature=True):
    return get_vgg("13", end_layer, pretrained, num_classes, only_feature)


def get_vgg16(end_layer=-1, pretrained=True, num_classes=1000, only_feature=True):
    return get_vgg("16", end_layer, pretrained, num_classes, only_feature)


def get_vgg19(end_layer=-1, pretrained=True, num_classes=1000, only_feature=True):
    return get_vgg("19", end_layer, pretrained, num_classes, only_feature)
