import sys
import time

import torch
from builder import export_from_registry
from core.utils.visualize import show_supported_models_on_command_line

from registry import model_registry
from core.utils.ckpt import CheckPoint
from scripts.detect import detect_video


# 权重文件位置
WEIGHTS = "saves/ultralytics/yolov8n_weights.pth"
# 输入文件类型：视频还是图片
TYPE = "image"  # "image" or "video"
# 测试图片路径的列表
IMAGE_PATHS = [
    "test/000000000049.jpg",
    "test/000000000139.jpg",
    "test/000000001584.jpg",
    "test/2007_000032.jpg",
    "test/2007_000033.jpg",
    "test/2007_002273.jpg",
]
# 原视频路径
SRC_VIDEO = "test/1.flv"
# 目标视频路径
DST_VIDEO = "test/det_1.mp4"


def main():
    t0 = time.time()
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

    assert TYPE in ["video", "image"], f"不支持{TYPE}类型的文件作为输入"
    if TYPE == "video":
        detect_video(
            model,
            src_video_path=SRC_VIDEO,
            dst_video_path=DST_VIDEO,
            decode_fn=model_object.predict,
        )
    else:
        for img in IMAGE_PATHS:
            model_object.predict(model, img, print_on=True, save_result=True)

    print(f"Total time: {(time.time() - t0):.2f}s")


if __name__ == "__main__":
    main()
