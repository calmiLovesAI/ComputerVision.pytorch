import os
from random import sample, shuffle

import cv2
import numpy as np
import xml.dom.minidom as xdom

from pycocotools.coco import COCO
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from configs.dataset_cfg import COCO_CFG, VOC_CFG
from core.utils.image_process import read_image, cv2_paste
from core.utils.useful_tools import get_random_number


class DetectionDataset(Dataset):
    def __init__(self, dataset_name: str, input_shape, mosaic, mosaic_prob, epoch_length, special_aug_ratio=0.7,
                 train=True):
        """
        :param dataset_name: 数据集名称, 'voc' or 'coco'
        :param input_shape: 输入图片大小, [h, w]
        :param mosaic: 是否开启mosaic数据增强
        :param mosaic_prob: 使用mosaic数据增强的概率
        :param epoch_length: 模型训练的epoch总数
        :param special_aug_ratio: 参考YoloX，由于Mosaic生成的训练图片，远远脱离自然图片的真实分布。
                                  当mosaic=True时，本代码会在special_aug_ratio范围内开启mosaic。
                                  默认为前70%个epoch，100个世代会开启70个世代。
        :param train: True表示训练集，False表示验证集
        """
        super().__init__()
        self.dataset_name = dataset_name.lower()
        assert self.dataset_name in ['voc', 'coco'], f"Unsupported dataset: {self.dataset_name}"
        self.input_shape = input_shape

        self.jitter = 0.3
        self.hue = 0.1
        self.sat = 0.7
        self.val = 0.4

        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.special_aug_ratio = special_aug_ratio
        self.epoch_length = epoch_length
        self.epoch_now = -1
        self.train = train

        if dataset_name == 'voc':
            self.voc_root, self.voc_class_names, self.voc_images, self.xml_paths, self.class2index = self._parse_voc(
                train)
        else:
            self.coco_images_root, self.coco_ids, self.coco = self._get_coco(train)

    def __len__(self):
        if self.dataset_name == "voc":
            return len(self.voc_images)
        elif self.dataset_name == "coco":
            return len(self.coco_ids)

    def __getitem__(self, item):
        if self.dataset_name == 'voc':
            if self.mosaic and get_random_number() < self.mosaic_prob and \
                    self.epoch_now < self.epoch_length * self.special_aug_ratio:
                image, box = self.mosaic_for_voc(item)
            else:
                # 使用opencv读取图片
                image_path = self.voc_images[item]
                image = read_image(image_path)

                # 解析xml
                box = self._parse_xml(self.xml_paths[item])
                box = np.array(box, dtype=np.float32)
                box = np.reshape(box, (-1, 5))  # shape: (N, 5)  N是这张图片包含的目标数

                image, box = self.get_random_data(image, box, random=self.train)



        else:
            if self.mosaic and get_random_number() < self.mosaic_prob and \
                    self.epoch_now < self.epoch_length * self.special_aug_ratio:
                # TODO mosaic for coco
                image, box = self.mosaic_for_coco(item)
            else:
                img_id = self.coco_ids[item]
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                target = self.coco.loadAnns(ann_ids)
                # 图片路径
                image_path = os.path.join(self.coco_images_root, self.coco.loadImgs(img_id)[0]['file_name'])
                assert os.path.exists(image_path), f"Image not exists: {image_path}"
                # 读取图片
                image = read_image(image_path)
                # 解析标注
                target = self._get_coco_bbox(target)
                target = np.array(target, dtype=np.float32)
                target = np.reshape(target, (-1, 5))

                image, box = self.get_random_data(image, target, random=self.train)

        # 归一化、通道交换
        image_tensor = TF.to_tensor(image)
        box = np.array(box, dtype=np.float32)
        # 对真实框进行预处理
        nL = len(box)
        labels_out = np.zeros((nL, 6))
        if nL:
            # ---------------------------------------------------#
            #   对真实框进行归一化，调整到0-1之间
            # ---------------------------------------------------#
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            # ---------------------------------------------------#
            #   序号为0、1的部分，为真实框的中心
            #   序号为2、3的部分，为真实框的宽高
            #   序号为4的部分，为真实框的种类
            #   [cx, cy, w, h, id]
            # ---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

            # ---------------------------------------------------#
            #   调整顺序，符合训练的格式
            #   labels_out中序号为0的部分在collect时处理
            # ---------------------------------------------------#
            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]

        # image: torch.Tensor, shape: (3, h, w)
        # labels_out: numpy.ndarray [nL, 6(默认0, id, cx, cy, w, h)]
        return image_tensor, labels_out

    def get_random_data(self, image, box, random=True):
        # 图像的高宽与目标高宽
        ih, iw, _ = image.shape
        h, w = self.input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = cv2.resize(src=image, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)
            new_image = np.full(shape=[h, w, 3], fill_value=128, dtype=np.uint8)
            new_image = cv2_paste(new_image, image, dx, dy)
            image_data = np.array(new_image, np.float32)

            # ---------------------------------#
            #   对真实框进行调整
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            return image_data, box

        # 对图像进行缩放并且扭曲长和宽
        new_ar = iw / ih * get_random_number(1 - self.jitter, 1 + self.jitter) / \
                 get_random_number(1 - self.jitter,
                                   1 + self.jitter)

        scale = get_random_number(0.4, 1.0)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = cv2.resize(src=image, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)
        # 将图像多余的部分加上灰条
        dx = int(get_random_number(0, w - nw))
        dy = int(get_random_number(0, h - nh))
        new_image = np.full(shape=[h, w, 3], fill_value=128, dtype=np.uint8)
        image = cv2_paste(new_image, image, dx, dy)

        # 翻转图像
        flip = get_random_number() < 0.5
        if flip:
            # 水平翻转
            image = cv2.flip(image, 1)

        image_data = np.array(image, dtype=np.uint8)
        # 对图像进行色域变换
        r = np.random.uniform(-1, 1, 3) * [self.hue, self.sat, self.val] + 1
        # 将图像转到HSV上
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        # 应用变换
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #  对真实框进行调整
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    def mosaic_body(self, image, box, input_h, input_w, index, min_offset_x, min_offset_y):
        # 图片的大小
        iw, ih, _ = image.shape

        # 是否翻转图片
        flip = get_random_number() < 0.5
        if flip and box.shape[0] > 0:
            # 水平翻转图片，box的值也相应改变
            image = cv2.flip(image, 1)
            box[:, [0, 2]] = iw - box[:, [2, 0]]

        # 对图像进行缩放并且扭曲长和宽
        new_ar = iw / ih * get_random_number(1 - self.jitter, 1 + self.jitter) / \
                 get_random_number(1 - self.jitter,
                                   1 + self.jitter)

        scale = get_random_number(0.4, 1.0)
        if new_ar < 1:
            nh = int(scale * input_h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * input_w)
            nh = int(nw / new_ar)
        image = cv2.resize(src=image, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)

        # 将图片放置在对应的位置上
        if index == 0:
            # 左上
            dx = int(input_w * min_offset_x) - nw
            dy = int(input_h * min_offset_y) - nh
        elif index == 1:
            # 左下
            dx = int(input_w * min_offset_x) - nw
            dy = int(input_h * min_offset_y)
        elif index == 2:
            # 右下
            dx = int(input_w * min_offset_x)
            dy = int(input_h * min_offset_y)
        elif index == 3:
            # 右上
            dx = int(input_w * min_offset_x)
            dy = int(input_h * min_offset_y) - nh

        # 创建一张背景颜色为(128, 128, 128)的RGB图像，大小为(input_h, input_w)
        new_image = np.full(shape=[input_h, input_w, 3], fill_value=128, dtype=np.uint8)
        # 把图片贴在这张“背景”的相应位置
        image_data = cv2_paste(new_image, image, dx, dy)

        box_data = []
        if box.shape[0] > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy

            # 限制坐标在0~input_w和0~input_h范围内
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > input_w] = input_w
            box[:, 3][box[:, 3] > input_h] = input_h

            # 求宽和高
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            # 保留宽、高都大于1个像素的box
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box

        return image_data, box_data

    def mosaic_for_voc(self, item):
        voc_ids = range(len(self.voc_images))
        # 从VOC2012中随机获取3个数据
        random_selected_items = sample(voc_ids, 3)
        # 加入当前item，一共是4张图片
        random_selected_items.append(item)
        # 打乱顺序
        shuffle(random_selected_items)
        # 网络输入图片的高、宽
        h, w = self.input_shape
        min_offset_x = get_random_number(0.3, 0.7)
        min_offset_y = get_random_number(0.3, 0.7)
        image_datas = []
        box_datas = []
        index = 0

        for image_id in random_selected_items:
            # 将图片读取为numpy.ndarray格式
            image = read_image(self.voc_images[image_id])

            # 获取此图片对对应的box
            box = self._parse_xml(self.xml_paths[image_id])
            box = np.array(box, dtype=np.float32)
            box = np.reshape(box, (-1, 5))  # shape: (N, 5)  N是这张图片包含的目标数

            image_data, box_data = self.mosaic_body(image, box, h, w, index, min_offset_x, min_offset_y)
            index += 1
            image_datas.append(image_data)
            box_datas.append(box_data)

        # 将4张图片放在一起
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)
        new_image = np.zeros(shape=[h, w, 3], dtype=np.uint8)
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        # 对图像进行色域变换
        r = np.random.uniform(-1, 1, 3) * [self.hue, self.sat, self.val] + 1
        # 将图像转到HSV上
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        # 应用变换
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    def mosaic_for_coco(self, item):
        # 从COCO中随机获取3个数据
        random_selected_items = sample(self.coco_ids, 3)
        # 加入当前item，一共是4张图片
        random_selected_items.append(self.coco_ids[item])
        # 打乱顺序
        shuffle(random_selected_items)
        # 网络输入图片的高、宽
        h, w = self.input_shape
        min_offset_x = get_random_number(0.3, 0.7)
        min_offset_y = get_random_number(0.3, 0.7)
        image_datas = []
        box_datas = []
        index = 0

        for img_id in random_selected_items:
            image_path = os.path.join(self.coco_images_root, self.coco.loadImgs(img_id)[0]['file_name'])
            # 将图片读取为numpy.ndarray格式
            image = read_image(image_path)

            # 获取此图片对对应的box
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            box = self._get_coco_bbox(target)
            box = np.array(box, dtype=np.float32)
            box = np.reshape(box, (-1, 5))   # shape: (N, 5)  N是这张图片包含的目标数

            image_data, box_data = self.mosaic_body(image, box, h, w, index, min_offset_x, min_offset_y)
            index += 1
            image_datas.append(image_data)
            box_datas.append(box_data)

        # 将4张图片放在一起
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)
        new_image = np.zeros(shape=[h, w, 3], dtype=np.uint8)
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        # 对图像进行色域变换
        r = np.random.uniform(-1, 1, 3) * [self.hue, self.sat, self.val] + 1
        # 将图像转到HSV上
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        # 应用变换
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    @staticmethod
    def _parse_voc(train=True):
        # VOC数据集的根目录和类别名
        voc_root, voc_class_names = VOC_CFG["root"], VOC_CFG["classes"]
        images_root = os.path.join(voc_root, "JPEGImages")
        if train:
            # 加载训练集
            train_txt = os.path.join(voc_root, "ImageSets", "Main", "train.txt")
            with open(train_txt, mode="r", encoding="utf-8") as f:
                image_names = f.read().strip().split('\n')
        else:
            # 加载验证集
            val_txt = os.path.join(voc_root, "ImageSets", "Main", "val.txt")
            with open(val_txt, mode="r", encoding="utf-8") as f:
                image_names = f.read().strip().split('\n')

        # 所有图片路径的列表
        image_paths = [os.path.join(images_root, f"{e}.jpg") for e in image_names]
        # 所有xml文件路径的列表
        xml_paths = [os.path.join(voc_root, "Annotations", f"{e}.xml") for e in image_names]
        # voc类别名的列表
        class2index = dict((v, k) for k, v in enumerate(voc_class_names))

        return voc_root, voc_class_names, image_paths, xml_paths, class2index

    def _get_coco(self, train=True):
        # coco数据集的根目录和类别名
        coco_root, coco_class_names = COCO_CFG["root"], COCO_CFG["classes"]
        mode = "train" if train else "val"
        images_root = os.path.join(coco_root, "images", f"{mode}2017")
        anno_file = os.path.join(coco_root, "annotations", f"instances_{mode}2017.json")
        coco = COCO(annotation_file=anno_file)
        ids = list(coco.imgToAnns.keys())  # 图片id列表
        class_to_id = dict(zip(coco_class_names, range(COCO_CFG["num_classes"])))
        class_to_coco_id = self._get_class_to_coco_id(coco.dataset["categories"])
        self.coco_id_to_class_id = dict([
            (class_to_coco_id[cls], class_to_id[cls])
            for cls in coco_class_names
        ])

        return images_root, ids, coco

    @staticmethod
    def _get_class_to_coco_id(categories):
        class_to_coco_id = dict()
        for category in categories:
            class_to_coco_id[category["name"]] = category["id"]
        return class_to_coco_id

    def _get_coco_bbox(self, target):
        bboxes = list()
        for obj in target:
            # (xmin, ymin, w, h)格式
            bbox = obj["bbox"]
            # 转为(xmin, ymin, xmax, ymax)格式
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            class_idx = self.coco_id_to_class_id[obj["category_id"]]
            bboxes.append([*bbox, class_idx])
        return bboxes

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
