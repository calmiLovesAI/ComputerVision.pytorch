from typing import List
import torch
import torchvision
from torch.utils.data import DataLoader
import random
import numpy as np
import torchvision.transforms
import torchvision.transforms.functional as F

from core.data.voc import VOCSegmentation


# the RGB value of the color corresponding to each category in the PASCAL VOC dataset
VOC_TABLE = {
    "background": (0, 0, 0),
    "aeroplane": (128, 0, 0),
    "bicycle": (0, 128, 0),
    "bird": (128, 128, 0),
    "boat": (0, 0, 128),
    "bottle": (128, 0, 128),
    "bus": (0, 128, 128),
    "car": (128, 128, 128),
    "cat": (64, 0, 0),
    "chair": (192, 0, 0),
    "cow": (64, 128, 0),
    "dining table": (192, 128, 0),
    "dog": (64, 0, 128),
    "horse": (192, 0, 128),
    "motorbike": (64, 128, 128),
    "person": (192, 128, 128),
    "potted plant": (0, 64, 0),
    "sheep": (128, 64, 0),
    "sofa": (0, 192, 0),
    "train": (128, 192, 0),
    "tv/monitor": (0, 64, 128),
}

CITYSCAPES_TABLE = {
    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "building": (70, 70, 70),
    "wall": (102, 102, 156),
    "fence": (190, 153, 153),
    "pole": (153, 153, 153),
    "traffic light": (250, 170, 30),
    "traffic sign": (220, 220, 0),
    "vegetation": (107, 142, 35),
    "terrain": (152, 251, 152),
    "sky": (70, 130, 180),
    "person": (220, 20, 60),
    "rider": (255, 0, 0),
    "car": (0, 0, 142),
    "truck": (0, 0, 70),
    "bus": (0, 60, 100),
    "train": (0, 80, 100),
    "motorcycle": (0, 0, 230),
    "bicycle": (119, 11, 32),
}


VOC_COLORMAP = [v for v in VOC_TABLE.values()]

VOC_CLASSES = [k for k in VOC_TABLE.keys()]

CITYSCAPES_COLORMAP = [v for v in CITYSCAPES_TABLE.values()]

CITYSCAPES_CLASSES = [k for k in CITYSCAPES_TABLE.keys()]


def label_indices(colormap, colormap2label):
    """
    将RGB值映射到对应的索引类别
    :param colormap:
    :param colormap2label:
    :return:
    """
    colormap = torch.from_numpy(colormap.astype(np.int64))
    idx = (colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2]
    return colormap2label[idx]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"     {t}"
        format_string += "\n"
        return format_string


class PIL2Numpy:
    def __call__(self, image, target):
        target = target.convert("RGB")
        return np.array(image), np.array(target)


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        target = F.resize(target.unsqueeze(dim=0), size=self.size)
        return F.resize(image, size=self.size), torch.squeeze(target, dim=0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class RGB2idx:
    def __init__(self, colormap2label):
        self.colormap2label = colormap2label

    def __call__(self, image, target):
        return image, label_indices(target, self.colormap2label)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomCrop:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, target):
        rect = torchvision.transforms.RandomCrop.get_params(
            image, (self.height, self.width)
        )
        image = torchvision.transforms.functional.crop(image, *rect)
        target = torchvision.transforms.functional.crop(target, *rect)
        return image, target

    def __repr__(self):
        return f"{self.__class__.__name__}(height={self.height}, width={self.width})"


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            tensor (Tensor): Tensor of label.
        Returns:
            Tensor: Normalized Tensor image.
            Tensor: Unchanged Tensor label
        """
        return F.normalize(image, mean=self.mean, std=self.std), target

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class RandomHorizontalFlip:
    """Horizontally flip the given Tensor randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        """
        Args:
            image (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(image), F.hflip(target)
        return image, target

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


def colormap2label(colormap):
    """
    构建从RGB到类别索引的映射
    :param colormap: list
    :return:
    """
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    for i, color in enumerate(colormap):
        colormap2label[(color[0] * 256 + color[1]) * 256 + color[2]] = i
    return colormap2label


def get_cityscapes_dataloader(root, batch_size, input_size, crop_size, is_train=True):
    base_size = input_size[1:]
    if isinstance(base_size, List):
        base_size = max(base_size)
        assert isinstance(base_size, int)
    cityscapes_colormap2label = colormap2label(CITYSCAPES_COLORMAP)
    if is_train:
        dataset = torchvision.datasets.Cityscapes(
            root=root,
            split="train",
            target_type="semantic",
            transforms=Compose(
                [
                    PIL2Numpy(),
                    ToTensor(),
                    RGB2idx(cityscapes_colormap2label),
                    Resize(size=base_size),
                    RandomCrop(*crop_size),
                    RandomHorizontalFlip(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )
        print(f"Loading train dataset with {len(dataset)} samples")
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = torchvision.datasets.Cityscapes(
            root=root,
            split="val",
            target_type="semantic",
            transforms=Compose(
                [
                    PIL2Numpy(),
                    ToTensor(),
                    RGB2idx(cityscapes_colormap2label),
                    Resize(size=tuple(crop_size)),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )
        print(f"Loading val dataset with {len(dataset)} samples")
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_voc_dataloader(root, batch_size, input_size, crop_size, is_train=True):
    base_size = input_size[1:]
    if isinstance(base_size, List):
        base_size = max(base_size)
        assert isinstance(base_size, int)
    voc_colormap2label = colormap2label(VOC_COLORMAP)
    if is_train:
        dataset = VOCSegmentation(
            root=root,
            image_set="train",
            transform=Compose(
                [
                    ToTensor(),
                    RGB2idx(voc_colormap2label),
                    Resize(size=base_size),
                    RandomCrop(*crop_size),
                    RandomHorizontalFlip(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )
        print(f"Loading train dataset with {len(dataset)} samples")
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = VOCSegmentation(
            root=root,
            image_set="val",
            transform=Compose(
                [
                    ToTensor(),
                    RGB2idx(voc_colormap2label),
                    Resize(size=tuple(crop_size)),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )
        print(f"Loading val dataset with {len(dataset)} samples")
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_sbd_dataloader(root, batch_size, input_size, crop_size, is_train=True):
    base_size = input_size[1:]
    if isinstance(base_size, List):
        base_size = max(base_size)
        assert isinstance(base_size, int)
    voc_colormap2label = colormap2label(VOC_COLORMAP)
    if is_train:
        dataset = torchvision.datasets.SBDataset(
            root=root,
            image_set="train",
            mode="segmentation",
            transforms=Compose(
                [
                    PIL2Numpy(),
                    ToTensor(),
                    RGB2idx(voc_colormap2label),
                    Resize(size=base_size),
                    RandomCrop(*crop_size),
                    RandomHorizontalFlip(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )
        print(f"Loading train dataset with {len(dataset)} samples")
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = torchvision.datasets.SBDataset(
            root=root,
            image_set="val",
            mode="segmentation",
            transforms=Compose(
                [
                    PIL2Numpy(),
                    ToTensor(),
                    RGB2idx(voc_colormap2label),
                    Resize(size=tuple(crop_size)),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )
        print(f"Loading val dataset with {len(dataset)} samples")
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
