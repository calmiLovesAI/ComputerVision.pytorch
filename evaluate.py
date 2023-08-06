import argparse

import torch
from builder import export_from_registry
from core.utils.visualize import show_supported_models_on_command_line
from registry import model_registry
from core.utils.ckpt import CheckPoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model name')
    parser.add_argument('--dataset', type=str, default='', help='the name of dataset')
    parser.add_argument('--ckpt', type=str, default='', help='model checkpoint path')
    opts = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg, model_class, _ = export_from_registry(opts.model)

    model_object = model_class(model_cfg, device)
    model, _ = model_object.build_model()
    model.to(device)

    # 加载模型权重
    CheckPoint.load_pure(opts.ckpt, device, model)
    print(f"Loaded weights: {opts.ckpt}")

    if opts.dataset == "voc":
        model_object.evaluate_on_voc(model, "result/voc", subset='val')
    elif opts.dataset == "coco":
        model_object.evaluate_on_coco(model, "result/coco", subset='val')
    else:
        raise ValueError(f"Unsupported dataset：{opts.dataset}")


if __name__ == '__main__':
    main()
