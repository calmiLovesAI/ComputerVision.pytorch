import os
from typing import List

import torch

from core.data.yolov3_dataloader import Yolo3Loader
from core.loss.yolov3_loss import YoloLoss, make_label
from core.metrics.eval import evaluate_pipeline
from core.models.yolov3_model import YoloV3
from core.predict.yolov3_decode import Decoder
from core.trainer.base import DetectionTrainer
from core.trainer.lr_scheduler import get_optimizer, warm_up_scheduler


class Yolo3Trainer(DetectionTrainer):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        # 损失函数的返回值要与这里的metrics_name一一对应
        self.metric_names = ["loss", "loc_loss", "conf_loss", "prob_loss"]
        # 是否在tqdm进度条中显示上述metrics
        self.show_option = [True, False, False, False]

    def initialize_model(self):
        self.model = YoloV3(self.cfg)
        self.model.to(device=self.device)
        self.model_name = "YOLOv3"

    def load_data(self):
        self.train_dataloader = Yolo3Loader(self.cfg,
                                            self.dataset_name,
                                            self.batch_size,
                                            self.input_image_size[1:]).__call__()

    def set_optimizer(self):
        self.optimizer = get_optimizer(self.optimizer_name, self.model, self.initial_lr)

    def set_lr_scheduler(self):
        self.lr_scheduler = warm_up_scheduler(optimizer=self.optimizer,
                                              warmup_epochs=self.warmup_epochs,
                                              multi_step=True,
                                              milestones=self.milestones,
                                              gamma=self.gamma,
                                              last_epoch=self.last_epoch)

    def set_criterion(self):
        self.criterion = YoloLoss(self.cfg, self.device)

    def train_loop(self, images, targets, scaler) -> List:
        images = images.to(device=self.device)
        targets = make_label(self.cfg, targets)
        targets = [x.to(device=self.device) for x in targets]

        self.optimizer.zero_grad()
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                preds = self.model(images)
                loss, loc_loss, conf_loss, prob_loss = self.criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            preds = self.model(images)
            loss, loc_loss, conf_loss, prob_loss = self.criterion(preds, targets)
            loss.backward()
            self.optimizer.step()

        return [loss, loc_loss, conf_loss, prob_loss]

    def evaluate(self,
                 weights=None,
                 subset='val',
                 skip=False):

        # 加载权重
        if weights is not None:
            self.load_weights(weights)
        # 切换为'eval'模式
        self.model.eval()

        evaluate_pipeline(model=self.model,
                          decoder=Decoder(self.cfg,
                                          conf_threshold=0.02,
                                          device=self.device),
                          input_image_size=self.input_image_size[1:],
                          map_out_root=os.path.join(self.result_path, "map"),
                          subset=subset,
                          device=self.device,
                          skip=skip)
