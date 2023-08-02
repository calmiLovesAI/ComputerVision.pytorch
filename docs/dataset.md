# 数据集

## COCO
从[这里](https://cocodataset.org/#download)下载2017 Train/Val images和2017 Train/Val annotations，
解压之后放在`${COCO_ROOT}`文件夹中，目录结构如下：
```
|-- coco
`-----|-- annotations
      |   |-- captions_train2017.json
      |   |-- captions_val2017.json
      |   |-- instances_train2017.json
      |   |-- instances_val2017.json
      |   |-- person_keypoints_train2017.json
      |   `-- person_keypoints_val2017.json
      |
       `-- images
            |-- train2017
            |   |-- ... 
            `-- val2017
                |-- ... 
```

## VOC2012
从[这里](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)下载，
解压之后放在`${VOC_ROOT}`文件夹中，目录结构如下：
```
|--VOCdevkit
      |---VOC2012
           |
            `-- |-- Annotations
                |-- ImageSets
                |-- JPEGImages
                |-- SegmentationClass
                `-- SegmentationObject
```
最后，修改`configs/dataset_cfg.py`文件中`VOC_CFG`和`COCO_CFG`中的`root`的值，分别为`${VOC_ROOT}/VOCdevkit/VOC2012/`和`${COCO_ROOT}/coco`