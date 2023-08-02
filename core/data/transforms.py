import random

import numpy as np
import torch
import torchvision.transforms.functional as F

from core.utils.image_process import letter_box


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), torch.from_numpy(target)


class Resize:
    def __init__(self, size):
        """
        :param size: list or tuple
        """
        self.size = size

    def __call__(self, image, target):
        image, scale, paddings = letter_box(image, self.size)
        top, bottom, left, right = paddings
        target[:, 0:-1] *= scale
        # xmin, xmax增加left像素
        target[:, 0:4:2] += left
        # 归一化
        target[:, 0:4:2] /= self.size[1]
        # ymin, ymax增加top像素
        target[:, 1:4:2] += top
        target[:, 1:4:2] /= self.size[0]
        return image, target


class ImageColorJitter:
    """
    随机改变一张图片的对比度、饱和度、亮度和色调
    """
    def __init__(self):
        # 对比度
        self.random_contrast_factor = random.uniform(0.5, 1.5)
        # 饱和度
        self.random_saturation_factor = random.uniform(0.8, 1.2)
        # 亮度
        self.random_brightness_factor = random.uniform(0.8, 1.2)
        # 色调
        self.random_hue_factor = random.uniform(-0.25, 0.25)

    def __call__(self, image, target):
        image = F.adjust_contrast(image, self.random_contrast_factor)
        image = F.adjust_saturation(image, self.random_saturation_factor)
        image = F.adjust_brightness(image, self.random_brightness_factor)
        image = F.adjust_hue(image, self.random_hue_factor)
        return image, target


class TargetPadding:
    def __init__(self, max_num_boxes):
        self.max_num_boxes = max_num_boxes

    def __call__(self, image, target):
        dst = np.full(shape=(self.max_num_boxes, 5), fill_value=-1, dtype=np.float32)
        for i in range(min(dst.shape[0], target.shape[0])):
            dst[i] = target[i]
        return image, dst
