import time
import numpy as np
import torch

from typing import List, Dict, Tuple


def get_current_format_time():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def get_format_filename(model_name: str, dataset_name: str, addition: str = None) -> str:
    return model_name + "_" + dataset_name + "_" + addition


def get_random_number(a: float = 0.0, b: float = 1.0):
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


def pbar_postfix_to_msg(postfix: Dict, is_value_str: bool = True) -> str:
    """
    将tqdm进度条的后缀信息转换为普通字符串
    :param postfix:
    :param is_value_str: True: 字典的value为字符串，False: 字典的value为数字
    :return:
    """
    if not is_value_str:
        return " ".join([f"{k}: {v:.5f}" for k, v in postfix.items()])
    return " ".join([f"{k}: {v}" for k, v in postfix.items()])


def check_list_slice_index_valid(n: int, start_idx: int, end_idx: int) -> bool:
    """
    判断列表的切片索引是否在合法范围内
    :param n: 列表长度
    :param start_idx: 起始索引，不支持负数
    :param end_idx: 结束索引，支持负数
    :return:
    """
    if start_idx < 0 or start_idx >= n:
        return False
    if end_idx >= n or end_idx < -n:
        return False
    return True
