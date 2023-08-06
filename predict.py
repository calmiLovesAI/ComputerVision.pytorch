import time
import argparse

import torch
from builder import export_from_registry
from core.utils.visualize import show_supported_models_on_command_line

from registry import model_registry
from core.utils.ckpt import CheckPoint
from scripts.detect import detect_video


# 测试图片路径的列表
IMAGE_PATHS = [
    "test/000000000049.jpg",
    "test/000000000139.jpg",
    "test/000000001584.jpg",
    "test/2007_000032.jpg",
    "test/2007_000033.jpg",
    "test/2007_002273.jpg",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='model name')
    parser.add_argument('--ckpt', type=str, default='', help='model checkpoint path')
    parser.add_argument('--type', type=str, default='', choices=["video", "image"], help='file type')
    parser.add_argument('--src', type=str, default='', help='source video path')
    parser.add_argument('--dst', type=str, default='', help='destination video path')
    opts = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg, model_class, _ = export_from_registry(opts.model)

    model_object = model_class(model_cfg, device)
    model, _ = model_object.build_model()
    model.to(device)

    # 加载模型权重
    CheckPoint.load_pure(opts.ckpt, device, model)
    print(f"Loaded weights: {opts.ckpt}")

    assert opts.type in ["video", "image"], f"不支持{opts.type}类型的文件作为输入"
    if opts.type == "video":
        detect_video(
            model,
            src_video_path=opts.src,
            dst_video_path=opts.dst,
            decode_fn=model_object.predict,
        )
    else:
        for img in IMAGE_PATHS:
            model_object.predict(model, img, print_on=True, save_result=True)

    print(f"Total time: {(time.time() - t0):.2f}s")


if __name__ == "__main__":
    main()
