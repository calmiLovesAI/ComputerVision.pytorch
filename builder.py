from check import check_model_name
from configs import *
from registry import config_registry, model_registry, trainer_registry
from core.trainer import *
from core.algorithms import *


def export_from_registry(name: str):
    """
    导出注册器中的值
    :param name: 注册器中的key
    :return: 配置文件类，模型类，训练类
    """
    name = name.lower()
    check_model_name(name)
    cfg_name = "cfg_" + name
    model_name = "model_" + name
    trainer_name = "trainer_" + name

    if cfg_name not in config_registry:
        raise KeyError(f"找不到{config_registry.name}注册器中的key：{cfg_name}")
    if model_name not in model_registry:
        raise KeyError(f"找不到{model_registry.name}注册器中的key：{model_name}")
    if trainer_name not in trainer_registry:
        raise KeyError(f"找不到{trainer_registry.name}注册器中的key：{trainer_name}")

    return (
        config_registry[cfg_name](),
        model_registry[model_name],
        trainer_registry[trainer_name],
    )
