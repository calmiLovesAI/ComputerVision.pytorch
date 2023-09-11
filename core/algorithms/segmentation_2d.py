import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np
import cv2
from tqdm import tqdm

from configs import DeeplabV3PlusConfig
from core.data.segmentation_dataset import CITYSCAPES_COLORMAP, VOC_COLORMAP, get_voc_dataloader
from core.loss.focal_loss import FocalLoss
from core.metrics.seg_metrics import SegmentationMetrics
from core.models.deeplabv3plus import DeeplabV3Plus
from core.utils.image_process import read_image, read_image_and_convert_to_tensor
from core.utils.useful_tools import move_to_device
from core.utils.visualize import now
from registry import model_registry


def postprocess_seg2d(dataset_type, pred, device):
    if dataset_type.lower() in ['voc', 'sbd']:
        colormap = torch.tensor(VOC_COLORMAP, device=device)
    elif dataset_type.lower() == 'cityscapes':
        colormap = torch.tensor(CITYSCAPES_COLORMAP, device=device)
    else:
        raise NotImplementedError(f"不支持{dataset_type}数据集")
    pred = torch.argmax(pred, dim=1)
    X = pred.long()
    return colormap[X, :]


def blend(foreground, background, weight):
    """
    将前景图像与背景图像进行融合
    """
    assert foreground.shape == background.shape
    foreground = np.uint8(foreground)
    blended = cv2.addWeighted(background, (1 - weight), foreground, weight, 0.0)
    return blended


@model_registry("deeplabv3plus")
class DeeplabV3PlusA:
    def __init__(self, cfg: DeeplabV3PlusConfig, device) -> None:
        self.cfg = cfg
        self.device = device

        self.loss_type = cfg.loss.loss_type
        self.num_classes = cfg.dataset.num_classes
        self.input_image_size = cfg.arch.input_size
        self.batch_size = cfg.train.batch_size
        self.dataset_name = cfg.dataset.dataset_name

    def build_model(self):
        return DeeplabV3Plus(num_classes=self.cfg.dataset.num_classes,
                             output_stride=self.cfg.arch.output_stride,
                             pretrained_backbone=self.cfg.arch.backbone_pretrained), "deeplabv3plus"

    def build_loss(self):
        if self.loss_type == "ce":
            criterion = nn.CrossEntropyLoss(reduction='mean')
        elif self.loss_type == "focal":
            criterion = FocalLoss()
        return criterion

    def _reshape_pred(self, pred, h, w):
        """
        将预测结果reshape到原图大小
        :param pred: 预测结果
        :param h: 原图高度
        :param w: 原图宽度
        """
        result = torch.squeeze(pred, dim=0)
        result = torch.permute(result, dims=[2, 0, 1])  # (h, w, c) -> (c, h, w)
        # back to original size
        result = F.resize(result, size=(h, w))
        result = torch.permute(result, dims=[1, 2, 0])  # (c, h, w) -> (h, w, c)
        return result

    def predict(self, model, image_path, print_on, save_result):
        """
        对单张图片进行预测
        :param model: 模型
        :param image_path: 图片路径
        :param print_on: 是否打印结果
        :param save_result: 是否保存结果
        """
        # 模型设为eval模式
        model.eval()
        # 处理单张图片
        original_img = read_image(image_path)
        image, h, w = read_image_and_convert_to_tensor(
            image_path, size=self.cfg.arch.input_size[1:], letterbox=False)
        image = image.to(self.device)

        with torch.no_grad():
            pred = model(image)
            pred = postprocess_seg2d(self.cfg.dataset.dataset_name, pred, self.device)  # shape: (1, h, w, 3)
        result = self._reshape_pred(pred, h, w)
        result = result.cpu().numpy()
        # blend foreground and background
        result = blend(foreground=result, background=original_img, weight=0.5)
        # rgb -> bgr
        result = result[..., ::-1]
        if save_result:
            save_dir = self.cfg.decode.test_results
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_filename = os.path.join(save_dir, os.path.basename(image_path).split(".")[0] + f"@{now()}.jpg")
            # 保存检测结果
            cv2.imwrite(save_filename, result)
        else:
            return result

    def evaluate_on_voc(self, model, results_out_root, subset='val'):
        """
        在voc数据集上验证结果
        :param model: 模型
        :param results_out_root: 结果保存路径
        :param subset: VOC的子集
        :return: None
        """
        model_name = "DeepLabV3Plus"
        results_out_root = os.path.join(results_out_root, model_name)
        if not os.path.exists(results_out_root):
            os.makedirs(results_out_root)
            print(f"创建路径{results_out_root}成功")
        results_filepath = os.path.join(results_out_root, f"{model_name}_{self.dataset_name}_{now()}.txt")
        model.eval()

        print(f"加载数据集VOC-{subset}......")
        dataset_root = self.cfg.dataset.root
        crop_size = self.cfg.arch.crop_size

        if subset == "val":
            val_dataloader = get_voc_dataloader(
                root=dataset_root,
                batch_size=self.batch_size,
                input_size=self.input_image_size,
                crop_size=crop_size,
                is_train=False,
            )
        else:
            raise ValueError(f"不支持VOC-{subset}")

        metrics = SegmentationMetrics(num_classes=self.num_classes)
        metrics.reset()
        print(f"验证中......")
        with tqdm(val_dataloader, desc="Evaluate") as pbar:
            with torch.no_grad():
                for i, (images, targets) in enumerate(pbar):
                    images = move_to_device(images, self.device)
                    targets = move_to_device(targets, self.device)
                    preds = model(images)

                    preds = torch.argmax(preds, dim=1)
                    metrics.add_batch(predictions=preds.cpu().numpy(), gts=targets.cpu().numpy())

        metric_results = metrics.get_results()
        formatted = (f"Overall Acc: {metric_results['Overall Acc']}\n"
                     f"Mean Acc: {metric_results['Mean Acc']}\n"
                     f"FreqW Acc: {metric_results['FreqW Acc']}\n"
                     f"Mean IoU: {metric_results['Mean IoU']}")
        print(f"结果：\n{formatted}")
        with open(file=results_filepath, mode="w", encoding="utf-8") as f:
            f.writelines(formatted)


def evaluate_on_coco(self, model, results_out_root, subset='val'):
    """
    在coco数据集上验证结果
    :param model: 模型
    :param results_out_root: 结果保存路径
    :param subset: coco的子集
    :return: None
    """
    raise NotImplementedError("不支持coco")
    pass
