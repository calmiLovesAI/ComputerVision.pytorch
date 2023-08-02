import sys

import torch
from builder import export_from_registry
from core.utils.visualize import show_supported_models_on_command_line
from registry import model_registry
from core.utils.ckpt import CheckPoint


# "voc" or "coco"
DATASET = "coco"
# 权重文件位置
WEIGHTS = "saves/CenterNet_coco_weights.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    show_supported_models_on_command_line(model_registry)
    print("请输入要训练的模型名：")
    model_name = sys.stdin.readline().strip()
    model_cfg, model_class, _ = export_from_registry(model_name)

    model_object = model_class(model_cfg, device)
    model, _ = model_object.build_model()
    model.to(device)

    # 加载模型权重
    CheckPoint.load_pure(WEIGHTS, device, model)
    print(f"Loaded weights: {WEIGHTS}")

    if DATASET == "voc":
        model_object.evaluate_on_voc(model, "result/voc", subset='val')
    elif DATASET == "coco":
        model_object.evaluate_on_coco(model, "result/coco", subset='val')
    else:
        raise ValueError(f"Unsupported dataset：{DATASET}")


if __name__ == '__main__':
    main()
