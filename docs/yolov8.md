# YOLOv8

## 安装和使用

1. 下载数据集：参考[数据集的配置](https://github.com/calmisential/Detection.pytorch/blob/main/docs/dataset.md)
2. 下载训练好的模型，使用`python convert_model.py`提取ultralytics的yolov8模型权重。
3. 验证模型的性能：验证yolov8模型在COCO数据集上的表现可以使用：
```
python evaluate.py --model yolo8_det --dataset coco --ckpt saves/ultralytics/yolov8n_weights.pth
```
1. 在图片（或视频）上测试：使用yolov8模型检测单张图片：
```
python predict.py --model yolo8_det --ckpt saves/ultralytics/yolov8n_weights.pth --type image 
```
1. 从头开始训练：训练yolov8检测模型可以使用：
```
python train.py --model yolo8_det
```
2. 恢复训练方法：修改模型`yolo8_det_cfg.py`配置文件中`self.resume_training`和`self.last_epoch`参数，然后运行`train.py`。

## 效果展示
![sample 1](https://github.com/calmiLovesAI/ComputerVision.pytorch/blob/main/performance/000000000139%402023-05-01-11-52-17.jpg) 
![sample_2](https://github.com/calmiLovesAI/ComputerVision.pytorch/blob/main/performance/000000001584%402023-05-01-11-52-17.jpg)
![sample_3](https://github.com/calmiLovesAI/ComputerVision.pytorch/blob/main/performance/000000006471%402023-05-01-11-52-17.jpg)

## 在不同数据集上的表现

