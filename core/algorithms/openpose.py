import torch
import torch.nn as nn

from core.models.vgg import get_vgg19


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        in_channels = c_in
        self.layers = nn.ModuleList()
        for _ in range(3):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.Conv2d(in_channels, c_out, kernel_size=3, stride=1, padding=1),
                nn.PReLU()
            ))
            in_channels = c_out

    def forward(self, x):
        x1 = self.layers[0](x)
        x2 = self.layers[1](x1)
        x3 = self.layers[2](x2)
        out = torch.cat((x1, x2, x3), dim=1)
        return out


class Stage0(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        in_channels = [c_in, 512, 512, 256]
        out_channels = [512, 512, 256, 256]
        self.convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.convs.append(nn.Conv2d(in_channels[i], out_channels[i], kernel_size=1, stride=1, padding=0))
        self.acts = nn.ModuleList([nn.PReLU() for _ in range(4)])

    def forward(self, x):
        for i in range(4):
            x = self.acts[i](self.convs[i](x))
        return x


class StageI(nn.Module):
    def __init__(self, c_in, c_1, c_2, act_fn):
        super().__init__()
        self.drops = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(5):
            self.drops.append(nn.Dropout())
            if i == 0:
                self.convs.append(ConvBlock(c_in, c_1))
            else:
                self.convs.append(ConvBlock(3 * c_1, c_1))
        self.conv6 = nn.Conv2d(3 * c_1, 512, kernel_size=1, stride=1, padding=0)
        self.act1 = nn.PReLU()
        self.conv7 = nn.Conv2d(512, c_2, kernel_size=1, stride=1, padding=0)
        self.act2 = act_fn

    def forward(self, inputs):
        if len(inputs) > 1:
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs[0]
        for i in range(5):
            x = self.drops[i](x)
            x = self.convs[i](x)
        x = self.act1(self.conv6(x))
        x = self.act2(self.conv7(x))
        return x


class CPM(nn.Module):
    def __init__(self, num_paf_filters=34, num_heatmap_filters=18):
        super().__init__()
        self.num_paf_filters = num_paf_filters
        self.num_heatmap_filters = num_heatmap_filters

        self.backbone = get_vgg19(end_layer=32)
        self.stage_0 = Stage0(c_in=512)
        self.stage_1 = StageI(c_in=256, c_1=96, c_2=self.num_paf_filters, act_fn=lambda x: x)
        num_channels = self.num_paf_filters + 256
        self.stage_2 = StageI(c_in=num_channels, c_1=128, c_2=self.num_paf_filters, act_fn=lambda x: x)
        self.stage_3 = StageI(c_in=num_channels, c_1=128, c_2=self.num_paf_filters, act_fn=lambda x: x)
        self.stage_4 = StageI(c_in=num_channels, c_1=128, c_2=self.num_paf_filters, act_fn=lambda x: x)
        self.stage_5 = StageI(c_in=num_channels, c_1=96, c_2=self.num_heatmap_filters, act_fn=torch.tanh)
        num_channels = 256 + self.num_paf_filters + self.num_heatmap_filters
        self.stage_6 = StageI(c_in=num_channels, c_1=128, c_2=self.num_heatmap_filters, act_fn=torch.tanh)

    def forward(self, x):
        """
        :param x: torch.Tensor, 假设shape为(batch, 3, 368, 368)
        :return:
        """
        x = self.backbone(x)  # (batch, 512, 46, 46)
        x = self.stage_0(x)  # (batch, 256, 46, 46)

        s1 = self.stage_1([x])  # (batch, n_paf, 46, 46)
        s2 = self.stage_2([s1, x])  # (batch, n_paf, 46, 46)
        s3 = self.stage_3([s2, x])  # (batch, n_paf, 46, 46)
        s4 = self.stage_4([s3, x])  # (batch, n_paf, 46, 46)
        s5 = self.stage_5([s4, x])  # (batch, n_heatmap, 46, 46)
        s6 = self.stage_6([s5, s4, x])  # (batch, n_heatmap, 46, 46)

        return s1, s2, s3, s4, s5, s6
