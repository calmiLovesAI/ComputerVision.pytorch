"""
The code is derived from：https://github.com/Tony-Y/pytorch_warmup
"""
import math

import torch.optim as optim
from contextlib import contextmanager


def get_warmup_params(warmup_period, group_count):
    if type(warmup_period) == list:
        if len(warmup_period) != group_count:
            raise ValueError(
                'size of warmup_period does not equal {}.'.format(group_count))
        for x in warmup_period:
            if type(x) != int:
                raise ValueError(
                    'An element in warmup_period, {}, is not an int.'.format(
                        type(x).__name__))
        warmup_params = [dict(warmup_period=x) for x in warmup_period]
    elif type(warmup_period) == int:
        warmup_params = [dict(warmup_period=warmup_period)
                         for _ in range(group_count)]
    else:
        raise TypeError('{} is not a list nor an int.'.format(
            type(warmup_period).__name__))
    return warmup_params


class Base:
    def __init__(self, optimizer, warmup_params, last_step=-1):
        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer
        self.last_step = last_step
        self.warmup_params = warmup_params
        self.lrs = [group['lr'] for group in optimizer.param_groups]
        self.dampen()

    def state_dict(self):
        """
        Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """
        Loads the warmup scheduler's state.
        :param state_dict:
        :return:
        """
        self.__dict__.update(state_dict)

    def dampen(self, step=None):
        """
        减小学习率
        :param step:  当前step
        :return:
        """
        if step is None:
            step = self.last_step + 1
        self.last_step = step

        for group, params in zip(self.optimizer.param_groups, self.warmup_params):
            omega = self.warmup_factor(step, **params)
            group['lr'] *= omega

    """
    contextmanager是Python中的一个装饰器，用于创建上下文管理器。
    上下文管理器是一种用于管理资源的对象，它定义了在进入和退出代码块时要执行的操作。
    contextmanager可以将一个普通的生成器函数转换为上下文管理器，从而使得使用上下文管理器更加方便。
    在使用with语句时，上下文管理器会在进入代码块之前执行__enter__方法，在退出代码块时执行__exit__方法。
    这样可以确保资源在使用完毕后被正确地释放。
    """

    @contextmanager
    def dampening(self):
        for group, lr in zip(self.optimizer.param_groups, self.lrs):
            group['lr'] = lr
        yield
        self.lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.dampen()

    def warmup_factor(self, step, **params):
        raise NotImplementedError


class LinearWarmup(Base):
    def __init__(self, optimizer, warmup_period, last_step=-1):
        """
        线性warmup
        :param optimizer: pytorch优化器
        :param warmup_period: list or int,
        :param last_step: 上一次step
        """

        warmup_params = get_warmup_params(warmup_period, len(optimizer.param_groups))
        super().__init__(optimizer, warmup_params, last_step)

    def warmup_factor(self, step, warmup_period):
        return min(1.0, (step + 1) / warmup_period)


class ExponentialWarmup(Base):
    def __init__(self, optimizer, warmup_period, last_step=-1):
        """
        指数warmup
        :param optimizer: pytorch优化器
        :param warmup_period: list or int,
        :param last_step: 上一次step
        """

        warmup_params = get_warmup_params(warmup_period, len(optimizer.param_groups))
        super().__init__(optimizer, warmup_params, last_step)

    def warmup_factor(self, step, warmup_period):
        return 1.0 - math.exp(-(step + 1) / warmup_period)


class UntunedLinearWarmup(LinearWarmup):
    """Untuned linear warmup schedule for Adam.

    This warmup scheme is described in
    `On the adequacy of untuned warmup for adaptive optimization
    <https://arxiv.org/abs/1910.04209>`_.

    Arguments:
        optimizer (Optimizer): an Adam optimizer
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, last_step=-1):
        def warmup_period_fn(beta2):
            return int(2.0 / (1.0 - beta2))

        warmup_period = [warmup_period_fn(x['betas'][1]) for x in optimizer.param_groups]
        super(UntunedLinearWarmup, self).__init__(optimizer, warmup_period, last_step)


class UntunedExponentialWarmup(ExponentialWarmup):
    """Untuned exponetial warmup schedule for Adam.

    This warmup scheme is described in
    `On the adequacy of untuned warmup for adaptive optimization
    <https://arxiv.org/abs/1910.04209>`_.

    Arguments:
        optimizer (Optimizer): an Adam optimizer
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, last_step=-1):
        def warmup_period_fn(beta2):
            return int(1.0 / (1.0 - beta2))

        warmup_period = [warmup_period_fn(x['betas'][1]) for x in optimizer.param_groups]
        super(UntunedExponentialWarmup, self).__init__(optimizer, warmup_period, last_step)
