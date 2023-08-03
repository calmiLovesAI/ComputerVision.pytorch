import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights


class UNet(nn.Module):
    """
    Part of the code derives from https://blog.csdn.net/smallworldxyl/article/details/121409052
    """

    def __init__(self, num_classes, in_channels, out_channels, pretrained=False):
        """
        :param num_classes: (int) - number of categories for classification
        :param pretrained: (boolean) - True means to use pretrained weights on ImageNet.
        """
        super(UNet, self).__init__()
        if pretrained:
            self.backbone = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        else:
            self.backbone = vgg16_bn(weights=None)
        del self.backbone.classifier
        del self.backbone.avgpool

        self.up4 = Up(in_channels[3], out_channels[3])
        self.up3 = Up(in_channels[2], out_channels[2])
        self.up2 = Up(in_channels[1], out_channels[1])
        self.up1 = Up(in_channels[0], out_channels[0])

        self.final = nn.Conv2d(in_channels=out_channels[0], out_channels=num_classes, kernel_size=1, stride=1,
                               padding=0)

    def forward(self, x):
        feat1 = self.backbone.features[:6](x)
        feat2 = self.backbone.features[6:13](feat1)
        feat3 = self.backbone.features[13:23](feat2)
        feat4 = self.backbone.features[23:33](feat3)
        feat5 = self.backbone.features[33:-1](feat4)

        up4 = self.up4(feat4, feat5)
        up3 = self.up3(feat3, up4)
        up2 = self.up2(feat2, up3)
        up1 = self.up1(feat1, up2)

        output = self.final(up1)

        return output


class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super(Up, self).__init__()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        output = torch.concat([x, self.upsampling(y)], dim=1)
        output = self.conv1(output)
        output = self.conv2(output)
        return output
