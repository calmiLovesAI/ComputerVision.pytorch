import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.centernet_cfg import CenternetConfig


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), stride=(stride, stride),
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=c_out)
        self.conv2 = nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=(3, 3), stride=(1, 1),
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=c_out)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class BottleNeck(nn.Module):
    expansion = 2

    def __init__(self, c_in, c_out, stride=1):
        super(BottleNeck, self).__init__()
        c_t = c_out // BottleNeck.expansion
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=c_t, kernel_size=(1, 1), stride=(1, 1),
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=c_t)
        self.conv2 = nn.Conv2d(in_channels=c_t, out_channels=c_t, kernel_size=(3, 3), stride=(stride, stride),
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=c_t)
        self.conv3 = nn.Conv2d(in_channels=c_t, out_channels=c_out, kernel_size=(1, 1), stride=(1, 1),
                               padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=c_out)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = F.relu(x + residual)
        return x


class BottleNeckX(nn.Module):
    cardinality = 32

    def __init__(self, c_in, c_out, stride=1):
        super(BottleNeckX, self).__init__()
        c_t = c_in * BottleNeckX.cardinality // 32
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=c_t, kernel_size=(1, 1), stride=(1, 1),
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=c_t)
        self.conv2 = nn.Conv2d(in_channels=c_t, out_channels=c_t, kernel_size=(3, 3), stride=(stride, stride),
                               padding=1, groups=BottleNeckX.cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=c_t)
        self.conv3 = nn.Conv2d(in_channels=c_t, out_channels=c_out, kernel_size=(1, 1), stride=(1, 1),
                               padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=c_out)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = F.relu(x + residual)
        return x


class Root(nn.Module):
    def __init__(self, c_in, c_out, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1), stride=(1, 1), padding=0,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=c_out)
        self.residual = residual

    def forward(self, inputs):
        x = self.bn(self.conv(torch.cat(inputs, dim=1)))
        if self.residual:
            x += inputs[0]
        x = F.relu(x)
        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride)
            self.tree2 = block(out_channels, out_channels, 1)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              root_residual=root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels

        if self.levels == 1:
            self.root = Root(root_dim, out_channels, root_residual)

        if stride > 1:
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, inputs, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(inputs) if self.downsample else inputs
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(inputs, residual=residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            outputs = self.root([x2, x1, *children])
        else:
            children.append(x1)
            outputs = self.tree2(x1, children=children)
        return outputs


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock,
                 residual_root=False, return_levels=False, pool_size=7):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes

        self.base_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=(7, 7), stride=(1, 1), padding=3,
                      bias=False),
            nn.BatchNorm2d(num_features=channels[0]),
            nn.ReLU(True)
        )

        self.level_0 = DLA._make_conv_level(channels[0], channels[0], levels[0])
        self.level_1 = DLA._make_conv_level(channels[0], channels[1], levels[1], 2)
        self.level_2 = Tree(levels=levels[2], block=block, in_channels=channels[1],
                            out_channels=channels[2], stride=2,
                            level_root=False, root_residual=residual_root)
        self.level_3 = Tree(levels=levels[3], block=block, in_channels=channels[2],
                            out_channels=channels[3], stride=2,
                            level_root=True, root_residual=residual_root)
        self.level_4 = Tree(levels=levels[4], block=block, in_channels=channels[3],
                            out_channels=channels[4], stride=2,
                            level_root=True, root_residual=residual_root)
        self.level_5 = Tree(levels=levels[5], block=block, in_channels=channels[4],
                            out_channels=channels[5], stride=2,
                            level_root=True, root_residual=residual_root)
        self.avgpool = nn.AvgPool2d(kernel_size=pool_size)
        self.final = nn.Conv2d(in_channels=channels[5], out_channels=num_classes, kernel_size=(1, 1), stride=(1, 1),
                               padding=0, bias=True)

    @staticmethod
    def _make_conv_level(in_channels, out_channels, convs, stride=1):
        layers = []
        for i in range(convs):
            layers.extend([
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=(3, 3),
                          stride=stride if i == 0 else 1,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(True)
            ])
        return nn.Sequential(*layers)

    def forward(self, x):
        y = list()
        x = self.base_layer(x)
        x = self.level_0(x)
        y.append(x.clone())
        x = self.level_1(x)
        y.append(x.clone())
        x = self.level_2(x)
        y.append(x.clone())
        x = self.level_3(x)
        y.append(x.clone())
        x = self.level_4(x)
        y.append(x.clone())
        x = self.level_5(x)
        y.append(x.clone())

        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.final(x)
            x = torch.reshape(x, (x.size()[0], -1))
            return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class IDAUp(nn.Module):
    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = nn.Sequential(
                    nn.Conv2d(in_channels=c,
                              out_channels=out_dim,
                              kernel_size=(1, 1),
                              stride=(1, 1),
                              padding=0,
                              bias=False),
                    nn.BatchNorm2d(num_features=out_dim),
                    nn.ReLU(True)
                )
            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.ConvTranspose2d(in_channels=out_dim, out_channels=out_dim, kernel_size=f * 2, stride=f,
                                        padding=f // 2, output_padding=0, groups=out_dim, bias=False)
            setattr(self, "proj_" + str(i), proj)
            setattr(self, "up_" + str(i), up)
        for i in range(1, len(channels)):
            node = nn.Sequential(
                nn.Conv2d(in_channels=out_dim * 2, out_channels=out_dim, kernel_size=node_kernel, stride=1,
                          padding=(node_kernel - 1) // 2, bias=False),
                nn.BatchNorm2d(num_features=out_dim),
                nn.ReLU(True)
            )
            setattr(self, "node_" + str(i), node)

    def forward(self, inputs):
        layers = list(inputs)
        for i, l in enumerate(layers):
            upsample = getattr(self, "up_" + str(i))
            project = getattr(self, "proj_" + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, "node_" + str(i))
            x = node(torch.cat([x, layers[i]], 1))
            y.append(x)
        return x, y


class DLAUp(nn.Module):
    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=np.int32)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(3, channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, inputs):
        layers = list(inputs)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])
            layers[-i - 1:] = y
        return x


class DLASeg(nn.Module):
    def __init__(self, base_name, heads, down_ratio=4, head_conv=256):
        super(DLASeg, self).__init__()
        self.heads = heads
        self.first_level = int(np.log2(down_ratio))
        self.base = DLASeg._get_base_block(base_name)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(in_channels=channels[self.first_level], out_channels=head_conv, kernel_size=(3, 3),
                              stride=(1, 1), padding=1, bias=True),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels=head_conv, out_channels=classes, kernel_size=(1, 1), stride=(1, 1), padding=0,
                              bias=True),
                )
            else:
                fc = nn.Conv2d(in_channels=channels[self.first_level], out_channels=classes, kernel_size=(1, 1),
                               stride=(1, 1), padding=0,
                               bias=True)
            self.__setattr__(head, fc)

    @staticmethod
    def _get_base_block(base_name):
        if base_name == "dla34":
            return DLA(levels=[1, 1, 1, 2, 2, 1], channels=[16, 32, 64, 128, 256, 512], block=BasicBlock,
                       return_levels=True)
        elif base_name == "dla60":
            return DLA(levels=[1, 1, 1, 2, 3, 1], channels=[16, 32, 128, 256, 512, 1024], block=BottleNeck,
                       return_levels=True)
        elif base_name == "dla102":
            return DLA(levels=[1, 1, 1, 3, 4, 1], channels=[16, 32, 128, 256, 512, 1024], block=BottleNeck,
                       residual_root=True, return_levels=True)
        elif base_name == "dla169":
            return DLA(levels=[1, 1, 2, 3, 5, 1], channels=[16, 32, 128, 256, 512, 1024], block=BottleNeck,
                       residual_root=True, return_levels=True)
        else:
            raise ValueError("The 'base_name' is invalid.")

    def forward(self, x):
        """
        :param x: torch.Tensor, shape: (N, C, H, W)
        :return: List of torch.Tensor, shape: [(N, num_classes, H/4, W/4), (N, 2, H/4, W/4), (N, 2, H/4, W/4)]
        """
        x = self.base(x)
        x = self.dla_up(x[self.first_level:])
        outputs = []
        for head in self.heads:
            outputs.append(self.__getattr__(head)(x))
        return outputs


class CenterNet(nn.Module):
    def __init__(self, cfg: CenternetConfig):
        super(CenterNet, self).__init__()
        self.heads = {"heatmap": cfg.dataset.num_classes, "wh": 2, "reg": 2}
        self.backbone = DLASeg(base_name="dla34", heads=self.heads)

    def forward(self, x):
        """
        :param x: torch.Tensor  shape: (N, 3, H, W)
        :return:  torch.Tensor  shape: (N, H/4, W/4, num_classes + 4)
        """
        x = self.backbone(x)
        x = torch.cat(tensors=x, dim=1)
        x = torch.permute(x, dims=(0, 2, 3, 1))
        return x
