import os
import traceback
import logging
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from core.utils.ckpt import CheckPoint
from core.utils.useful_tools import get_format_filename, get_current_format_time, pbar_postfix_to_msg


# 装饰器，用于给模型添加预训练权重
def use_pretrained_model(init_fn):
    def wrapper(*args, **kwargs):
        ret = init_fn(*args, **kwargs)
        self = args[0]
        if self.pretrained and CheckPoint.check(self.pretrained_weights):
            CheckPoint.load_pure(path=self.pretrained_weights,
                                 device=self.device,
                                 model=self.model)
            print(f"Load pretrained weights from {self.pretrained_weights}.")

        return ret

    return wrapper


class MeanMetric:

    def __init__(self):
        self.accumulated = 0
        self.count = 0

    def update(self, value):
        self.accumulated += value
        self.count += 1

    def result(self):
        return self.accumulated / self.count

    def reset(self):
        self.__init__()


class BaseTrainer:

    def __init__(self, cfg, device, set_epoch=False):
        self.cfg = cfg
        self.device = device
        self.set_epoch = set_epoch

        # 模型总的训练轮数
        self.total_epoch = cfg.train.epoch
        # 恢复训练时的上一次epoch是多少，-1表示从epoch=0开始训练
        self.last_epoch = cfg.train.last_epoch
        # 恢复训练时加载的checkpoint文件
        self.resume_training_weights = cfg.train.resume_training
        # 数据集名称
        self.dataset_name = cfg.dataset.dataset_name
        # 数据集中的目标类别数，对于voc，这个值是20，对于coco，它是80
        self.num_classes = cfg.dataset.num_classes
        # 模型输入图片大小 (C, H, W)
        self.input_image_size = cfg.arch.input_size
        self.batch_size = cfg.train.batch_size

        # 优化器
        self.optimizer_name = cfg.optimizer.name
        self.warmup_iters = cfg.train.warmup_iters
        self.initial_lr = cfg.train.initial_lr
        self.milestones = cfg.train.milestones
        self.gamma = cfg.train.gamma
        self.num_workers = cfg.train.num_workers
        self.eval_interval = cfg.train.eval_interval
        self.save_interval = cfg.train.save_interval
        self.save_path = cfg.train.save_path

        # 是否使用预训练模型
        self.pretrained = cfg.train.pretrained
        self.pretrained_weights = cfg.train.pretrained_weights
        # 自动创建目录用于存放保存的模型
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.result_path = cfg.decode.test_results
        # 自动创建目录用于存放检测结果
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        self.tensorboard_on = cfg.train.tensorboard_on
        self.mixed_precision = cfg.train.mixed_precision
        self.writer = None

        # 训练集
        self.train_dataloader = None
        # 验证集
        self.val_dataloader = None
        # 模型
        self.model = None
        # 模型名称
        self.model_name = None

        # 优化器
        self.optimizer = None
        # 学习率调整策略
        self.lr_scheduler = None
        # warmup策略
        self.warmup_scheduler = None
        # 损失函数
        self.criterion = None
        self.model_algorithm = None

        self.metric_names = []
        self.show_option = []

        self.set_model_algorithm()
        self.load_data()
        # 总迭代次数
        self.last_iter = (self.last_epoch + 1) * len(self.train_dataloader)
        self.milestones = [(m + 1) * len(self.train_dataloader) for m in self.milestones]
        self.initialize_model()
        self.set_optimizer()
        self.set_lr_scheduler()
        self.set_criterion()

        # 日志系统
        self.train_logger = logging.getLogger("TRAIN")
        train_logger_file = os.path.join(cfg.log.root, get_format_filename(model_name=self.model_name,
                                                                           dataset_name=self.dataset_name,
                                                                           addition=get_current_format_time() + ".log"))
        if not os.path.exists(cfg.log.root):
            os.makedirs(cfg.log.root)
        handler = logging.FileHandler(filename=train_logger_file, encoding="utf-8")
        self.train_logger.setLevel(logging.INFO)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        self.train_logger.addHandler(handler)

        self.log_print_interval = cfg.log.print_interval

    def set_model_algorithm(self):
        return None

    def initialize_model(self):
        pass

    def load_data(self):
        pass

    def set_optimizer(self):
        pass

    def set_lr_scheduler(self):
        pass

    def set_criterion(self):
        pass

    def train_loop(self, batch_data, scaler) -> List:
        return []

    def train(self):
        n = len(self.metric_names)
        train_metrics = [MeanMetric() for _ in range(n)]
        if self.tensorboard_on:
            writer = SummaryWriter()
            try:
                writer.add_graph(
                    self.model,
                    torch.randn(self.batch_size,
                                *self.input_image_size,
                                dtype=torch.float32,
                                device=self.device))
            except Exception:
                traceback.print_exc()

        if CheckPoint.check(self.resume_training_weights):
            # 从checkpoint恢复训练
            CheckPoint.load(path=self.resume_training_weights,
                            device=self.device,
                            model=self.model,
                            pure=False,
                            optimizer=self.optimizer,
                            scheduler=self.lr_scheduler,
                            warm_up=self.warmup_scheduler)
            print(
                f"After loading weights from {self.resume_training_weights}, it will resume training from epoch-{self.last_epoch}."
            )

        if self.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        # 打印训练超参数信息
        self.train_logger.info(
            msg=f"\n模型: {self.model_name}\n"
                f"数据集: {self.dataset_name}\n"
                f"batch_size: {self.batch_size}\n"
                f"训练总epoch: {self.total_epoch}\n"
                f"起始epoch: {self.last_epoch + 1}\n"
                f"优化器: {self.optimizer_name}\n"
                f"初始学习率: {self.initial_lr}\n"
                f"输入图片尺寸: {self.input_image_size[1]}x{self.input_image_size[2]}"
        )
        # 打印模型网络结构
        self.train_logger.info(
            msg=f"网络结构: \n{self.model}"
        )

        for epoch in range(self.last_epoch + 1, self.total_epoch):
            # 切换为训练模式
            self.model.train()
            # 重置
            for m in train_metrics:
                m.reset()

            if self.set_epoch:
                # 设置当前epoch
                self.train_dataloader.dataset.epoch_now = epoch

            with tqdm(self.train_dataloader,
                      desc=f"Epoch-{epoch}/{self.total_epoch}") as pbar:
                for i, batch_data in enumerate(pbar):
                    metrics = self.train_loop(batch_data, scaler)
                    assert len(metrics) == n
                    for j in range(n):
                        train_metrics[j].update(metrics[j].item())
                    # 当前学习率
                    current_lr = self.optimizer.state_dict(
                    )['param_groups'][0]['lr']
                    if self.tensorboard_on:
                        writer.add_scalar(
                            tag="Learning rate",
                            scalar_value=current_lr,
                            global_step=epoch * len(self.train_dataloader) + i)
                        for k in range(n):
                            writer.add_scalar(
                                tag=f"Train/{self.metric_names[k]}",
                                scalar_value=metrics[k].item(),
                                global_step=epoch * len(self.train_dataloader)
                                            + i)

                    # 设置进度条后缀
                    postfix_info = {}
                    for p in range(n):
                        if self.show_option[p]:
                            postfix_info[self.metric_names[
                                p]] = f"{(train_metrics[p].result()):.5f}"
                    pbar.set_postfix(postfix_info)

                    # 打印日志信息
                    if i % self.log_print_interval == 0:
                        self.train_logger.info(
                            msg=f"Epoch: {epoch}/{self.total_epoch}, step: {i}/{len(self.train_dataloader)}, "
                                f"{pbar_postfix_to_msg(postfix_info)}")

                    if self.warmup_scheduler:
                        with self.warmup_scheduler.dampening():
                            self.lr_scheduler.step()

            if self.eval_interval != 0 and epoch % self.eval_interval == 0:
                evaluation = self.evaluate_loop()
                print([f"{k}={v:.5f}" for k, v in evaluation.items()])
                if self.tensorboard_on:
                    for k, v in evaluation.items():
                        writer.add_scalar(
                            tag=f"Val/{k}",
                            scalar_value=v,
                            global_step=epoch)
                self.train_logger.info(
                    msg=f"===========Evaluate after epoch-{epoch}============\n {pbar_postfix_to_msg(evaluation, False)}")

            if epoch % self.save_interval == 0 or epoch == self.total_epoch - 1:
                CheckPoint.save(
                    model=self.model,
                    path=Path(self.save_path).joinpath(
                        f"{self.model_name}_{self.dataset_name.lower()}_epoch-{epoch}.pth"
                    ),
                    optimizer=self.optimizer,
                    scheduler=self.lr_scheduler,
                    warm_up=self.warmup_scheduler)
        if self.tensorboard_on:
            writer.close()
        # 保存最终模型
        CheckPoint.save(
            model=self.model,
            path=Path(self.save_path).joinpath(
                f"{self.model_name}_{self.dataset_name.lower()}_final.pth"))

    def evaluate_loop(self) -> Dict:
        return {}
