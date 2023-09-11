from typing import List, Dict

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from core.algorithms.yolo_v7 import YOLOv7

from core.data.collate import yolo7_collate
from core.data.detection_dataset import DetectionDataset
from core.trainer.base import BaseTrainer
from core.trainer.lr_scheduler import get_optimizer, warm_up_scheduler, EnhancedMultiStepLR
from core.trainer.warm_up import LinearWarmup
from core.utils.useful_tools import move_to_device

from registry import trainer_registry
from configs import Yolo7Config


@trainer_registry("yolo7")
class Yolo7Trainer(BaseTrainer):
    def __init__(self, cfg: Yolo7Config, device):
        super().__init__(cfg, device, True)
        self.cfg = cfg
        self.device = device
        # 损失函数的返回值要与这里的metrics_name一一对应
        self.metric_names = ["loss", "box_loss", "obj_loss", "cls_loss"]
        # 是否在tqdm进度条中显示上述metrics
        self.show_option = [True, True, True, True]

    def set_model_algorithm(self):
        self.model_algorithm = YOLOv7(self.cfg, self.device)

    def initialize_model(self):
        self.model, self.model_name = self.model_algorithm.build_model()
        self.model.to(device=self.device)

    def load_data(self):
        train_dataset = DetectionDataset(dataset_name=self.dataset_name,
                                         input_shape=self.input_image_size[1:],
                                         mosaic=True,
                                         mosaic_prob=0.5,
                                         epoch_length=self.total_epoch,
                                         special_aug_ratio=0.7,
                                         train=True)
        val_dataset = DetectionDataset(dataset_name=self.dataset_name,
                                       input_shape=self.input_image_size[1:],
                                       mosaic=False,
                                       mosaic_prob=0,
                                       epoch_length=self.total_epoch,
                                       special_aug_ratio=0,
                                       train=False)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                           shuffle=True, num_workers=self.num_workers,
                                           drop_last=True, collate_fn=yolo7_collate)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size,
                                         shuffle=True, num_workers=self.num_workers,
                                         drop_last=True, collate_fn=yolo7_collate)

    def set_optimizer(self):
        self.optimizer = get_optimizer(self.optimizer_name, self.model, self.initial_lr)

    def set_lr_scheduler(self):
        self.lr_scheduler = EnhancedMultiStepLR(optimizer=self.optimizer,
                                                milestones=self.milestones,
                                                gamma=self.gamma,
                                                last_epoch=self.last_iter)
        if self.warmup_iters > 0:
            self.warmup_scheduler = LinearWarmup(
                optimizer=self.optimizer,
                warmup_period=self.warmup_iters,
                last_step=self.last_iter,
            )

    def set_criterion(self):
        self.criterion = self.model_algorithm.build_loss()

    def train_loop(self, batch_data, scaler) -> List:
        images = move_to_device(batch_data[0], self.device)
        targets = move_to_device(batch_data[1], self.device)

        self.optimizer.zero_grad()
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                preds = self.model(images)
                loss, box_loss, obj_loss, cls_loss = self.criterion(preds, targets, images)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            preds = self.model(images)
            loss, box_loss, obj_loss, cls_loss = self.criterion(preds, targets, images)
            loss.backward()
            self.optimizer.step()

        return [loss, box_loss, obj_loss, cls_loss]

    def evaluate_loop(self) -> Dict:
        self.model.eval()
        val_loss = 0
        num_batches = len(self.val_dataloader)

        with tqdm(self.val_dataloader, desc="Evaluate") as pbar:
            with torch.no_grad():
                for i, (images, targets) in enumerate(pbar):
                    images = images.to(device=self.device)
                    targets = targets.to(device=self.device)
                    preds = self.model(images)
                    loss_value, _, _, _ = self.criterion(preds, targets, images)

                    val_loss += loss_value.item()

        val_loss /= num_batches
        return {'val_loss': val_loss}
