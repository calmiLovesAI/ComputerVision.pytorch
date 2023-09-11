# ComputerVision.pytorch
计算机视觉

## 安装
确保安装以下版本的开发环境：
- cuda: 11.6
- torch: 1.13.1
- torchvision: 0.14.1

在终端运行
```commandline
git clone https://github.com/calmisential/DeepLearning.pytorch.git
cd DeepLearning.pytorch
pip install -r requirements.txt
```

## 使用方法
### 1. 数据集
[数据集的配置](https://github.com/calmiLovesAI/ComputerVision.pytorch/blob/main/docs/dataset.md)
### 2. 模型训练和预测
- [YOLOv8](https://github.com/calmiLovesAI/ComputerVision.pytorch/blob/main/docs/yolov8.md)


### 3. Tensorboard使用
1. 在`configs`目录下的`xxx_cfg.py`配置文件中，确保`self.tensorboard_on`的值为**True**
2. 在项目根目录下，运行
```commandline
tensorboard --logdir=runs
```
打开浏览器窗口查看训练过程中的记录，可以使用`--log_prefix="prefix_1"`选择名字以prefix_1开头的文件

## 参考
- https://github.com/bubbliiiing/ssd-pytorch
- https://github.com/Megvii-BaseDetection/YOLOX
- https://github.com/bubbliiiing/yolov7-pytorch
- https://github.com/ultralytics/ultralytics
- https://github.com/lufficc/SSD