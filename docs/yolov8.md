# YOLOv8

## 安装和使用

1. 下载数据集：参考[数据集的配置](https://github.com/calmisential/Detection.pytorch/blob/main/docs/dataset.md)
2. 下载训练好的模型
3. 验证模型的性能：修改`evaluate.py`中的配置参数，验证模型在VOC或COCO数据集上的表现。
4. 在图片（或视频）上测试：修改`detect.py`中的配置参数，然后运行`detect.py`。
5. 从头开始训练：修改`train.py`中的配置参数，然后运行`train.py`。
6. 恢复训练方法：修改模型`yolo8_det_cfg.py`配置文件中`self.resume_training`和`self.last_epoch`参数，然后运行`train.py`。

## 效果展示
![sample 1](https://github.com/calmisential/DeepLearning.pytorch/blob/main/performance/000000000139%402023-05-01-11-52-17.jpg) 
![sample_2](https://github.com/calmisential/DeepLearning.pytorch/blob/main/performance/000000001584%402023-05-01-11-52-17.jpg)
![sample_3](https://github.com/calmisential/DeepLearning.pytorch/blob/main/performance/000000006471%402023-05-01-11-52-17.jpg)

## 在不同数据集上的表现

