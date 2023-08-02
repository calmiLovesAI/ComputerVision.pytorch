import torch
import torch.nn as nn


class DarknetConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1)):
        super(DarknetConv2d, self).__init__()
        if stride == (2, 2):
            self.padding_strategy = nn.ZeroPad2d(padding=(1, 0, 1, 0))
        else:
            padding_length = (kernel_size[0] - 1) // 2
            self.padding_strategy = nn.ZeroPad2d(
                padding=(padding_length, padding_length, padding_length, padding_length))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=0)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        x = self.padding_strategy(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = DarknetConv2d(in_channels=n_channels, out_channels=n_channels // 2, kernel_size=(1, 1),
                                   stride=(1, 1))
        self.conv2 = DarknetConv2d(in_channels=n_channels // 2, out_channels=n_channels, kernel_size=(3, 3),
                                   stride=(1, 1))

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv1 = DarknetConv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.s1 = self._make_stage(num_blocks=1, in_channels=32, out_channels=64)
        self.s2 = self._make_stage(num_blocks=2, in_channels=64, out_channels=128)
        self.s3 = self._make_stage(num_blocks=8, in_channels=128, out_channels=256)
        self.s4 = self._make_stage(num_blocks=8, in_channels=256, out_channels=512)
        self.s5 = self._make_stage(num_blocks=4, in_channels=512, out_channels=1024)

    def _make_stage(self, num_blocks, in_channels, out_channels):
        x = list()
        x.append(DarknetConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(2, 2)))
        for _ in range(num_blocks):
            x.append(ResidualBlock(n_channels=out_channels))
        return nn.Sequential(*x)

    def forward(self, x):
        x = self.conv1(x)
        x = self.s1(x)
        x = self.s2(x)
        o3 = self.s3(x)
        o2 = self.s4(o3)
        o1 = self.s5(o2)
        return o1, o2, o3


class YoloBlock(nn.Module):
    def __init__(self, in_channels, n_channels, final_out_channels):
        super(YoloBlock, self).__init__()
        self.conv1 = DarknetConv2d(in_channels=in_channels, out_channels=n_channels, kernel_size=(1, 1))
        self.conv2 = DarknetConv2d(in_channels=n_channels, out_channels=2 * n_channels, kernel_size=(3, 3))
        self.conv3 = DarknetConv2d(in_channels=2 * n_channels, out_channels=n_channels, kernel_size=(1, 1))
        self.conv4 = DarknetConv2d(in_channels=n_channels, out_channels=2 * n_channels, kernel_size=(3, 3))
        self.conv5 = DarknetConv2d(in_channels=2 * n_channels, out_channels=n_channels, kernel_size=(1, 1))

        self.conv6 = DarknetConv2d(in_channels=n_channels, out_channels=2 * n_channels, kernel_size=(3, 3))
        self.normal_conv = nn.Conv2d(in_channels=2 * n_channels, out_channels=final_out_channels, kernel_size=(1, 1),
                                     stride=(1, 1), padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        o1 = x
        x = self.conv6(x)
        o2 = self.normal_conv(x)
        return o1, o2


class YoloV3(nn.Module):
    def __init__(self, cfg):
        super(YoloV3, self).__init__()
        num_classes = cfg.arch.num_classes
        self.out_channels = (num_classes + 5) * 3
        self.backbone = Darknet53()
        self.block1 = YoloBlock(in_channels=1024, n_channels=512, final_out_channels=self.out_channels)
        self.block2 = YoloBlock(in_channels=768, n_channels=256, final_out_channels=self.out_channels)
        self.block3 = YoloBlock(in_channels=384, n_channels=128, final_out_channels=self.out_channels)

        self.conv1 = DarknetConv2d(in_channels=512, out_channels=256, kernel_size=(1, 1))
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv2 = DarknetConv2d(in_channels=256, out_channels=128, kernel_size=(1, 1))
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        """

        :param x: Tensor, shape: (batch_size, 3, 416, 416)
        :return: List of Tensor: [(batch_size, 255, 13, 13), (batch_size, 255, 26, 26), (batch_size, 255, 52, 52)]
        """
        y1, y2, y3 = self.backbone(x)

        y1, o1 = self.block1(y1)
        y1 = self.conv1(y1)
        y1 = self.upsample1(y1)

        y2 = torch.cat([y2, y1], dim=1)
        y2, o2 = self.block2(y2)
        y2 = self.conv2(y2)
        y2 = self.upsample2(y2)

        y3 = torch.cat([y2, y3], dim=1)
        _, o3 = self.block3(y3)
        return o1, o2, o3

    def get_model_name(self):
        return "YoloV3"

