import os
import numpy as np
import xml.dom.minidom as xdom

from torch.utils.data import Dataset

from core.data import get_voc_root_and_classes
from core.utils.image_process import read_image


class Voc(Dataset):
    def __init__(self, train=True, transforms=None):
        super().__init__()
        # VOC数据集的根目录和类别名
        self.root, self.class_names = get_voc_root_and_classes()
        # 对(image, target)的变换
        self.transforms = transforms
        xmls_root = os.path.join(self.root, "Annotations")
        images_root = os.path.join(self.root, "JPEGImages")
        images = self._load_train_val_data(train=train)
        # 所有图片路径的列表
        self.image_paths = [os.path.join(images_root, f"{e}.jpg") for e in images]
        # 所有xml文件路径的列表
        self.xml_paths = [os.path.join(xmls_root, f"{e}.xml") for e in images]
        # voc类别名的列表
        self.class2index = dict((v, k) for k, v in enumerate(self.class_names))

    def _load_train_val_data(self, train=True):
        if train:
            # 加载训练集
            train_txt = os.path.join(self.root, "ImageSets", "Main", "train.txt")
            with open(train_txt, mode="r", encoding="utf-8") as f:
                image_names = f.read().strip().split('\n')
            return image_names
        else:
            # 加载验证集
            val_txt = os.path.join(self.root, "ImageSets", "Main", "val.txt")
            with open(val_txt, mode="r", encoding="utf-8") as f:
                image_names = f.read().strip().split('\n')
            return image_names

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        # 获取第item个图片的路径和它对应的标签文件的路径
        xml_path = self.xml_paths[item]
        image_path = self.image_paths[item]
        image = read_image(image_path)

        target = self._parse_xml(xml_path)
        target = np.array(target, dtype=np.float32)
        target = np.reshape(target, (-1, 5))  # shape: (N, 5)  N是这张图片包含的目标数
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target

    def _parse_xml(self, xml):
        box_class_list = []
        DOMTree = xdom.parse(xml)
        annotation = DOMTree.documentElement
        obj = annotation.getElementsByTagName("object")
        for o in obj:
            o_list = []
            obj_name = o.getElementsByTagName("name")[0].childNodes[0].data
            bbox = o.getElementsByTagName("bndbox")[0]
            xmin = bbox.getElementsByTagName("xmin")[0].childNodes[0].data
            ymin = bbox.getElementsByTagName("ymin")[0].childNodes[0].data
            xmax = bbox.getElementsByTagName("xmax")[0].childNodes[0].data
            ymax = bbox.getElementsByTagName("ymax")[0].childNodes[0].data
            o_list.append(float(xmin))
            o_list.append(float(ymin))
            o_list.append(float(xmax))
            o_list.append(float(ymax))
            o_list.append(self.class2index[obj_name])
            box_class_list.append(o_list)
        # [[xmin, ymin, xmax, ymax, class_index], ...]
        return box_class_list
