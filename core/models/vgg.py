import warnings

import torch
import torch.nn as nn

from core.utils.useful_tools import check_list_slice_index_valid

__all__ = [
    "BaseVGG",
    "get_vgg11",
    "get_vgg13",
    "get_vgg16",
    "get_vgg19"
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
            if v == 'M':  # 池化层
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:  # 卷积层
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


def get_vgg11(start_layer=0, end_layer=-1, pretrained=True, num_classes=1000):
    """
    导出VGG11模型
    :param start_layer: int, 开始层的序号，默认为0
    :param end_layer: int, 结束层的序号，默认为-1
    :param pretrained: bool，默认为True
    :param num_classes: int, 最后一个Linear层输出通道数，默认为1000
    :return:
    """
    vgg11 = BaseVGG("11", 3, num_classes=num_classes)
    n_layers = len(vgg11)
    print(vgg11)
    if not check_list_slice_index_valid(n_layers, start_idx=start_layer, end_idx=end_layer):
        raise ValueError(f"vgg11有{n_layers}层，但是start_layer = {start_layer}，end_layer = {end_layer}")
    if pretrained:
        pass


class VGG11(BaseVGG):
    model_name = "VGG11"

    def __init__(self, cfg):
        super(VGG11, self).__init__(cfg,
                                    [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
                                    True)

        default_shape = (224, 224)
        input_shape = tuple(cfg["Train"]["input_size"][1:])
        if input_shape != default_shape:
            warnings.warn(
                "你正在使用的输入图片大小：{}与{}默认的输入图片大小：{}不符！".format(input_shape, self.model_name,
                                                                                   default_shape))


class VGG13(BaseVGG):
    model_name = "VGG13"

    def __init__(self, cfg):
        super(VGG13, self).__init__(cfg,
                                    [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
                                    True)

        default_shape = (224, 224)
        input_shape = tuple(cfg["Train"]["input_size"][1:])
        if input_shape != default_shape:
            warnings.warn(
                "你正在使用的输入图片大小：{}与{}默认的输入图片大小：{}不符！".format(input_shape, self.model_name,
                                                                                   default_shape))


class VGG16(BaseVGG):
    model_name = "VGG16"

    def __init__(self, cfg):
        super(VGG16, self).__init__(cfg,
                                    [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512,
                                     "M"],
                                    True)
        default_shape = (224, 224)
        input_shape = tuple(cfg["Train"]["input_size"][1:])
        if input_shape != default_shape:
            warnings.warn(
                "你正在使用的输入图片大小：{}与{}默认的输入图片大小：{}不符！".format(input_shape, self.model_name,
                                                                                   default_shape))


class VGG19(BaseVGG):
    model_name = "VGG19"

    def __init__(self, cfg):
        super(VGG19, self).__init__(cfg,
                                    [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512,
                                     512, 512, 512, "M"],
                                    True)

        default_shape = (224, 224)
        input_shape = tuple(cfg["Train"]["input_size"][1:])
        if input_shape != default_shape:
            warnings.warn(
                "你正在使用的输入图片大小：{}与{}默认的输入图片大小：{}不符！".format(input_shape, self.model_name,
                                                                                   default_shape))
