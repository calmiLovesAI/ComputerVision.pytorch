MODELS = [
    "yolo7",
    "yolo8_det",
    "ssd",
    "centernet",
    "deeplabv3plus",
]


def check_model_name(name: str):
    if name not in MODELS:
        raise ValueError(f"暂不支持模型：{name}\n你可以输入下列模型之一：\n{MODELS}")
    else:
        pass
