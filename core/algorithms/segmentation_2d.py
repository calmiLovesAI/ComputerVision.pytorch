import torch
import torch.nn as nn
import numpy as np

from configs import DeeplabV3PlusConfig
from core.loss.focal_loss import FocalLoss
from core.models.deeplabv3plus import DeeplabV3Plus
from registry import model_registry


@model_registry("deeplabv3plus")
class DeeplabV3PlusA:
    def __init__(self, cfg: DeeplabV3PlusConfig, device) -> None:
        self.cfg = cfg
        self.device = device

        self.loss_type = cfg.loss.loss_type

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
    


