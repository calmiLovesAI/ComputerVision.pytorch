import time
import numpy as np
import torch


def get_current_format_time():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def get_format_filename(model_name: str, dataset_name: str, addition: str = None) -> str:
    return model_name + "_" + dataset_name + "_" + addition


def get_random_number(a=0.0, b=1.0):
    """生成[a,b)范围内的随机数"""
    return np.random.rand() * (b - a) + a


def move_to_device(t, device):
    """
    把数据t移动到设备device上
    :param t:
    :param device:
    :return:
    """
    # t是tensor
    if isinstance(t, torch.Tensor):
        return t.to(device)
    # t是list或者tuple
    elif isinstance(t, (list, tuple)):
        return [move_to_device(v, device) for v in t]
    # t是dict
    elif isinstance(t, dict):
        return {k: move_to_device(v, device) for k, v in t.items()}
    # t是其他类型
    else:
        return t


def pbar_postfix_to_msg(postfix: dict, is_value_str=True):
    """
    将tqdm进度条的后缀信息转换为普通字符串
    :param postfix:
    :param is_value_str: True: 字典的value为字符串，False: 字典的value为数字
    :return:
    """
    if not is_value_str:
        return " ".join([f"{k}: {v:.5f}" for k, v in postfix.items()])
    return " ".join([f"{k}: {v}" for k, v in postfix.items()])
    