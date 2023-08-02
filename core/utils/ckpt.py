import os

import numpy as np
import torch


class CheckPoint:
    @staticmethod
    def check(path):
        """
        判断权重文件是否存在
        :param path:
        :return:
        """
        if path is None:
            return False
        return os.path.exists(path)

    @staticmethod
    def load_pretrained(model, weights):
        assert CheckPoint.check(weights), f"The pretrained model weights {weights} does not exist."
        model_dict = model.state_dict()
        print(f"Loading pretrained model state dict from {weights}...")
        pretrained_dict = torch.load(weights)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # 显示匹配上和没有匹配上的key
        print(f"Successfully loaded {len(load_key)} keys, they are: {load_key[:20]}...")
        print(f"Failed to load {len(no_load_key)} keys, they are: {no_load_key[:20]}...")

    @staticmethod
    def save(model, path, optimizer=None, scheduler=None, warm_up=None):
        if optimizer is None and scheduler is None:
            # 仅保存模型的state_dict
            torch.save(model.state_dict(), path)
        else:
            obj = {"model": model.state_dict()}
            if optimizer is not None:
                obj["optimizer"] = optimizer.state_dict()
            if scheduler is not None:
                obj["scheduler"] = scheduler.state_dict()
            if warm_up is not None:
                obj["warm_up"] = warm_up.state_dict()
            torch.save(obj, path)

    @staticmethod
    def load(path, device, model, pure=False, optimizer=None, scheduler=None, warm_up=None):
        ckpt = torch.load(path, map_location=device)
        if pure:
            # ckpt中仅保存了模型的state_dict
            model.load_state_dict(ckpt)
        else:
            model.load_state_dict(ckpt["model"])
            if optimizer is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            if warm_up is not None:
                scheduler.load_state_dict(ckpt["warm_up"])
        del ckpt

    @staticmethod
    def load_pure(path, device, model):
        ckpt = torch.load(path, map_location=device)
        if "model" in ckpt.keys():
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt)
        del ckpt