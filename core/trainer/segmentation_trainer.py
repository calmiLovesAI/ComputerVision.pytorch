
from typing import Dict, List
import torch
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from core.algorithms.segmentation_2d import DeeplabV3PlusA
from core.data.segmentation_dataset import (
    get_cityscapes_dataloader,
    get_sbd_dataloader,
    get_voc_dataloader,
)
from core.metrics.seg_metrics import SegmentationMetrics
from core.trainer.base import BaseTrainer
from core.trainer.lr_scheduler import get_optimizer, warm_up_scheduler
from core.trainer.warm_up import LinearWarmup
from core.utils.useful_tools import move_to_device
from registry import trainer_registry
from configs import DeeplabV3PlusConfig


@trainer_registry("deeplabv3plus")
class DeeplabV3PlusTrainer(BaseTrainer):
    def __init__(self, cfg: DeeplabV3PlusConfig, device):
        super().__init__(cfg, device, False)
        self.cfg = cfg
        self.device = device

        self.metrics = SegmentationMetrics(num_classes=self.num_classes)
        # 损失函数的返回值要与这里的metrics_name一一对应
        self.metric_names = ["loss"]
        # 是否在tqdm进度条中显示上述metrics
        self.show_option = [True]

    def set_model_algorithm(self):
        self.model_algorithm = DeeplabV3PlusA(self.cfg, self.device)

    def initialize_model(self):
        self.model, self.model_name = self.model_algorithm.build_model()
        self.model.to(device=self.device)

    def load_data(self):
        dataset_root = self.cfg.dataset.root
        crop_size = self.cfg.arch.crop_size
        match self.dataset_name.lower():
            case "voc":
                self.train_dataloader = get_voc_dataloader(
                    root=dataset_root,
                    batch_size=self.batch_size,
                    input_size=self.input_image_size,
                    crop_size=crop_size,
                    is_train=True,
                )
                self.val_dataloader = get_voc_dataloader(
                    root=dataset_root,
                    batch_size=self.batch_size,
                    input_size=self.input_image_size,
                    crop_size=crop_size,
                    is_train=False,
                )
            case "cityscapes":
                self.train_dataloader = get_cityscapes_dataloader(
                    root=dataset_root,
                    batch_size=self.batch_size,
                    input_size=self.input_image_size,
                    crop_size=crop_size,
                    is_train=True,
                )
                self.val_dataloader = get_cityscapes_dataloader(
                    root=dataset_root,
                    batch_size=self.batch_size,
                    input_size=self.input_image_size,
                    crop_size=crop_size,
                    is_train=False,
                )
            case "sbd":
                self.train_dataloader = get_sbd_dataloader(
                    root=dataset_root,
                    batch_size=self.batch_size,
                    input_size=self.input_image_size,
                    crop_size=crop_size,
                    is_train=True,
                )
                self.val_dataloader = get_sbd_dataloader(
                    root=dataset_root,
                    batch_size=self.batch_size,
                    input_size=self.input_image_size,
                    crop_size=crop_size,
                    is_train=False,
                )
            case _:
                raise NotImplementedError
        
    def set_optimizer(self):
        self.optimizer = get_optimizer(self.optimizer_name, self.model, self.initial_lr)
    
    def set_lr_scheduler(self):
        self.lr_scheduler = MultiStepLR(
            optimizer=self.optimizer,
            milestones=self.milestones,
            gamma=self.gamma,
            last_epoch=self.last_iter,
        )
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
                loss = self.criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            preds = self.model(images)
            loss = self.criterion(preds, targets)
            loss.backward()
            self.optimizer.step()

        return [loss]
    
    def evaluate_loop(self) -> Dict:
        self.model.eval()
        val_loss = 0.0
        num_batches = len(self.val_dataloader)
        self.metrics.reset()

        with tqdm(self.val_dataloader, desc="Evaluate") as pbar:
            with torch.no_grad():
                for i, (images, targets) in enumerate(pbar):
                    images = move_to_device(images, self.device)
                    targets = move_to_device(targets, self.device)
                    preds = self.model(images)
                    loss_value = self.criterion(preds, targets)

                    preds = torch.argmax(preds, dim=1)
                    self.metrics.add_batch(predictions=preds.cpu().numpy(), gts=targets.cpu().numpy())

                    val_loss += loss_value.item()

        val_loss /= num_batches
        metric_results = self.metrics.get_results()
        return {"val_loss": val_loss, **metric_results}
