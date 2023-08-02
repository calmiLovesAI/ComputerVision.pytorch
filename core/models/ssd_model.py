import torch
import torch.nn as nn
from configs.ssd_cfg import SsdConfig


class VGG(nn.Module):
    def __init__(self, batch_norm=False, pretrained=True):
        super().__init__()
        self.batch_norm = batch_norm
        model_params = [64, 64, "M", 128, 128, "M", 256, 256, 256, "C",
                        512, 512, 512, "M", 512, 512, 512]
        in_channels = 3
        layers = []
        for v in model_params:
            if v == 'M':  # 池化层
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == "C":  # 池化层, ceil_mode=True
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:  # 卷积层
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
                if batch_norm:  # BN
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        layers += [
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 1024, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        ]

        self.layers = nn.ModuleList(layers)
        if pretrained:
            self._load_pretrained("https://download.pytorch.org/models/vgg16_bn-6c64b313.pth")

    def _load_pretrained(self, url):
        state_dict = torch.hub.load_state_dict_from_url(url)
        state_dict = {k.replace('features.', ''): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)
        print("Loaded ImageNet1K weights")

    def forward(self, x):
        """
        :param x: torch.Tensor, shape: (N, 3, 300, 300)
        :return: List of torch.Tensor, shape: [torch.Size([N, 512, 38, 38]), torch.Size([N, 1024, 19, 19])]
        """
        extract_index = 32 if self.batch_norm else 22  # 需要提取的特征所在的层的索引
        outputs = []
        for i, l in enumerate(self.layers):
            x = l(x)
            if i == extract_index:
                outputs.append(x)
        outputs.append(x)
        return outputs


class ExtraLayer(nn.Module):
    def __init__(self, c_in, type="300"):
        super(ExtraLayer, self).__init__()
        assert type in ["300", "512"], f"Invalid type: {type}"
        self.conv1 = nn.Conv2d(c_in, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        if type == "300":
            self.conv5 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
            self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)

            self.conv7 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
            self.conv8 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)

            self.conv9 = None
            self.conv10 = None
        else:
            self.conv5 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
            self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

            self.conv7 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
            self.conv8 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

            self.conv9 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
            self.conv10 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        x = self.conv2(x)
        outputs.append(x)  # (batch, 512, 10, 10)
        x = self.conv3(x)
        x = self.conv4(x)
        outputs.append(x)  # (batch, 256, 5, 5)
        x = self.conv5(x)
        x = self.conv6(x)
        outputs.append(x)  # (batch, 256, 3, 3)
        x = self.conv7(x)
        x = self.conv8(x)
        outputs.append(x)  # (batch, 256, 1, 1)

        if self.conv9 is not None and self.conv10 is not None:
            x = self.conv9(x)
            x = self.conv10(x)
            outputs.append(x)

        return outputs


class L2Normalize(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Normalize, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class SSD(nn.Module):
    def __init__(self, cfg: SsdConfig):
        super(SSD, self).__init__()
        self.num_classes = cfg.dataset.num_classes + 1
        self.input_size = cfg.arch.input_size[1]
        # 每个stage分支输出的feature map中每个像素位置处的先验框数量
        self.num_boxes_per_pixel = [len(ar) + 1 for ar in cfg.arch.aspect_ratios]
        self.feature_channels = cfg.arch.feature_channels
        self.backbone = VGG(batch_norm=True, pretrained=cfg.train.pretrained)
        self.l2_norm = L2Normalize(n_channels=512, scale=20)
        self.extras = ExtraLayer(c_in=1024, type=str(self.input_size))
        self.locs, self.confs = self._make_locs_and_confs()

    def _make_locs_and_confs(self):
        loc_layers = nn.ModuleList()  # 回归层
        conf_layers = nn.ModuleList()  # 分类层
        for i in range(len(self.feature_channels)):
            loc_layers.append(
                nn.Conv2d(
                    in_channels=self.feature_channels[i],
                    out_channels=self.num_boxes_per_pixel[i] * 4,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ))
            conf_layers.append(
                nn.Conv2d(
                    in_channels=self.feature_channels[i],
                    out_channels=self.num_boxes_per_pixel[i] * self.num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
        return loc_layers, conf_layers

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        x1, x2 = self.backbone(x)
        x1 = self.l2_norm(x1)
        sources.append(x1)
        sources.append(x2)
        # o1, o2, o3, o4 = self.extras(x2)
        sources.extend(self.extras(x2))

        for (x, l, c) in zip(sources, self.locs, self.confs):
            loc.append(l(x))
            conf.append(c(x))

        # shape: (batch, 34928)
        loc = torch.cat(tensors=[torch.reshape(o, shape=(o.size(0), -1)) for o in loc], dim=1)
        # shape: (batch, 183372)
        conf = torch.cat(tensors=[torch.reshape(o, shape=(o.size(0), -1)) for o in conf], dim=1)

        loc = torch.reshape(loc, shape=(loc.shape[0], -1, 4))  # (batch, 8732, 4)
        conf = torch.reshape(conf, shape=(conf.shape[0], -1, self.num_classes))  # (batch, 8732, self.num_classes)

        return loc, conf
