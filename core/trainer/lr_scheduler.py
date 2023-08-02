import math
from copy import deepcopy

import torch


def warm_up_scheduler(optimizer, warmup_epochs, multi_step=True, milestones=None, gamma=None, last_epoch=-1):
    """
    warmup: 训练开始时从一个较小的学习率逐渐上升到初始学习率
    :param optimizer:  优化器
    :param warmup_epochs:  warmup的epoch数量
    :param multi_step: 是否使用MultiStepLR学习率衰减
    :param milestones:   MultiStepLR中的milestones参数
    :param gamma:   MultiStepLR中的gamma参数
    :param last_epoch:  -1表示从epoch-0开始
    :return:
    """

    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            if not multi_step:
                return 1.0
            else:
                assert milestones is not None and gamma is not None, "milestones and gamma can't be None"
                factor = 1.0
                for m in milestones:
                    if (epoch + 1) > m:
                        factor *= gamma
                return factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda, last_epoch=last_epoch)


def get_optimizer(optimizer_name, model, initial_lr):
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam([{"params": model.parameters(),
                                       'initial_lr': initial_lr}], lr=initial_lr)
    else:
        raise ValueError(f"{optimizer_name} is not supported")
    return optimizer


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """
    指数移动平均（Exponential Moving Average）也叫权重移动平均（Weighted Moving Average），
    是一种给予近期数据更高权重的平均方法。
    """
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
