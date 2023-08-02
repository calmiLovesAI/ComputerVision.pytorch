import json
import os

import torch
import numpy as np
import xml.etree.ElementTree as ET

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from configs.dataset_cfg import VOC_CFG, COCO_CFG
from core.loss.centernet_loss import CombinedLoss, RegL1Loss
from core.metrics.mAP import get_map, get_coco_map
from core.models import CenterNet
from core.utils.bboxes import xywh_to_xyxy, truncate_array, xywh_to_xyxy_torch
from core.utils.gaussian import gaussian_radius, draw_umich_gaussian
from core.utils.image_process import read_image_and_convert_to_tensor, read_image, reverse_letter_box
from core.utils.nms import diou_nms
from core.utils.visualize import show_detection_results

from registry import model_registry
from configs import CenternetConfig


@model_registry("centernet")
class CenterNetA:
    def __init__(self, cfg: CenternetConfig, device):
        self.cfg = cfg
        self.device = device

        # 类别数目
        self.num_classes = cfg.dataset.num_classes

        # 损失函数中的权重分配
        self.hm_weight = cfg.loss.hm_weight
        self.wh_weight = cfg.loss.wh_weight
        self.off_weight = cfg.loss.off_weight

        # 每张图片中最多的目标数目
        self.max_num_boxes = cfg.train.max_num_boxes

        # 输入图片的尺寸
        self.input_size = cfg.arch.input_size[1:]
        # 特征图的下采样倍数
        self.downsampling_ratio = cfg.arch.downsampling_ratio
        # 特征图的尺寸 [h, w]
        self.feature_size = [self.input_size[0] // self.downsampling_ratio,
                             self.input_size[1] // self.downsampling_ratio]

        self.K = cfg.decode.max_boxes_per_img
        self.conf_threshold = cfg.decode.score_threshold
        self.nms_threshold = cfg.decode.nms_threshold
        self.use_nms = cfg.decode.use_nms

        self.letterbox_image = cfg.decode.letterbox_image

    def build_model(self):
        model = CenterNet(self.cfg)
        model_name = "CenterNet"
        return model, model_name

    def build_loss(self):
        return CombinedLoss(self.num_classes, self.hm_weight, self.wh_weight, self.off_weight)

    def generate_targets(self, label):
        """
        :param label: numpy.ndarray, shape: (N, 6(_, class_id, cx, cy, w, h))
        :return:
        """
        class_label = label[:, 1:2]
        # 坐标由(cx, cy, w, h)转换为(xmin, ymin, xmax, ymax)
        coord_label = xywh_to_xyxy(label[:, 2:])
        # shape: (N, 5(xmin, ymin, xmax, ymax, class_id))
        label = np.concatenate((coord_label, class_label), axis=-1)
        # 确保label的第一个维度是max_num_boxes
        label = truncate_array(label, self.max_num_boxes, False)
        hm = np.zeros((self.feature_size[0], self.feature_size[1], self.num_classes), dtype=np.float32)
        reg = np.zeros((self.max_num_boxes, 2), dtype=np.float32)
        wh = np.zeros((self.max_num_boxes, 2), dtype=np.float32)
        reg_mask = np.zeros((self.max_num_boxes,), dtype=np.float32)
        ind = np.zeros((self.max_num_boxes,), dtype=np.float32)

        for j, item in enumerate(label):
            # 坐标映射到特征图尺寸上
            item[:4:2] = item[:4:2] * self.feature_size[1]
            item[1:4:2] = item[1:4:2] * self.feature_size[0]
            xmin, ymin, xmax, ymax, class_id = item
            # 类别id
            class_id = class_id.astype(np.int32)
            # 目标框的宽高
            h, w = int(ymax - ymin), int(xmax - xmin)
            # 高斯半径
            radius = gaussian_radius((h, w))
            radius = max(0, int(radius))
            # 中心点坐标
            ctr_x, ctr_y = (xmin + xmax) / 2, (ymin + ymax) / 2
            center_point = np.array([ctr_x, ctr_y], dtype=np.float32)
            center_point_int = center_point.astype(np.int32)
            _hm = draw_umich_gaussian(hm[:, :, class_id], center_point_int, radius)
            hm[:, :, class_id] = _hm

            reg[j] = center_point - center_point_int
            wh[j] = np.array([w, h], dtype=np.float32)
            reg_mask[j] = 1
            ind[j] = center_point_int[1] * self.feature_size[1] + center_point_int[0]

        # 返回torch.Tensor
        return torch.from_numpy(hm), torch.from_numpy(reg), torch.from_numpy(wh), torch.from_numpy(
            reg_mask), torch.from_numpy(ind)

    def predict(self, model, image_path, print_on, save_result):
        model.eval()
        # 处理单张图片
        image, h, w = read_image_and_convert_to_tensor(
            image_path, size=self.cfg.arch.input_size[1:], letterbox=self.letterbox_image)
        image = image.to(self.device)

        with torch.no_grad():
            preds = model(image)
            # shape: (N, 4), (N,), (N,)
            boxes, scores, classes = self.decode_boxes(preds, h, w)

        if boxes.shape[0] == 0:
            print(f"No object detected")
            return read_image(image_path, mode='bgr')

        return show_detection_results(image_path=image_path,
                                      dataset_name=self.cfg.dataset.dataset_name,
                                      boxes=boxes,
                                      scores=scores,
                                      class_indices=classes,
                                      print_on=print_on,
                                      save_result=save_result,
                                      save_dir=self.cfg.decode.test_results)

    def evaluate_on_voc(self, model, map_out_root, subset='val'):
        # 切换为'eval'模式
        model.eval()
        gt_path = os.path.join(map_out_root, 'ground-truth')
        detections_path = os.path.join(map_out_root, 'detection-results')
        images_optional_path = os.path.join(map_out_root, 'images-optional')

        for p in [map_out_root, gt_path, detections_path, images_optional_path]:
            if not os.path.exists(p):
                os.makedirs(p)

        voc_root = VOC_CFG["root"]
        voc_class_names = VOC_CFG["classes"]
        if subset == 'val':
            image_ids = open(os.path.join(voc_root, "ImageSets", "Main", "val.txt"), mode='r').read().strip().split(
                '\n')
        elif subset == 'test':
            image_ids = open(os.path.join(voc_root, "ImageSets", "Main", "test.txt"), mode='r').read().strip().split(
                '\n')
        else:
            raise ValueError(f"sub_set must be one of 'test' and 'val', but got {subset}")

        with tqdm(image_ids, desc=f"Evaluate on voc-{subset}") as pbar:
            for image_id in pbar:
                # 图片预处理
                image_path = os.path.join(voc_root, "JPEGImages", f"{image_id}.jpg")
                image, h, w = read_image_and_convert_to_tensor(image_path, size=self.cfg.arch.input_size[1:],
                                                               letterbox=self.letterbox_image)
                image = image.to(self.device)

                with torch.no_grad():
                    preds = model(image)
                    boxes, scores, classes = self.decode_boxes(preds, h, w, conf_threshold=0.001)

                if boxes.shape[0] == 0:
                    # 填充0
                    boxes = np.zeros(shape=(1, 4), dtype=np.float32)
                    scores = np.zeros(shape=(1,), dtype=np.float32)
                    classes = np.zeros(shape=(1,), dtype=np.int32)

                # 将检测结果写入txt文件中
                with open(file=os.path.join(detections_path, f"{image_id}.txt"), mode='w', encoding='utf-8') as f:
                    for i, c in enumerate(classes):
                        predicted_class = voc_class_names[int(c)]
                        score = str(scores[i])

                        top = boxes[i, 1]  # ymin
                        left = boxes[i, 0]  # xmin
                        bottom = boxes[i, 3]  # ymax
                        right = boxes[i, 2]  # xmax

                        f.write(f"{predicted_class} {score[:6]} {int(left)} {int(top)} {int(right)} {int(bottom)}\n")

        print("Successfully generated detection results")

        print("Generating ground truth")
        for image_id in tqdm(image_ids):
            with open(file=os.path.join(gt_path, f"{image_id}.txt"), mode='w', encoding='utf-8') as gt_f:
                root = ET.parse(os.path.join(voc_root, "Annotations", f"{image_id}.xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') is not None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in voc_class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        gt_f.write(f"{obj_name} {left} {top} {right} {bottom} difficult\n")
                    else:
                        gt_f.write(f"{obj_name} {left} {top} {right} {bottom}\n")

        print("Calculating metrics")
        # 第一个参数表示预测框与真实框的重合程度
        get_map(0.5, draw_plot=True, score_threshold=0.5, path=map_out_root)
        print("Calculating coco metrics")
        get_coco_map(class_names=voc_class_names, path=map_out_root)

    def evaluate_on_coco(self, model, map_out_root, subset='val'):
        if subset == 'val':
            # 在coco val2017上测试
            cocoGt_path = os.path.join(COCO_CFG["root"], "annotations", "instances_val2017.json")
            dataset_img_path = os.path.join(COCO_CFG["root"], "images", "val2017")

        # 切换为'eval'模式
        model.eval()
        if not os.path.exists(map_out_root):
            os.makedirs(map_out_root)
        cocoGt = COCO(cocoGt_path)
        ids = list(cocoGt.imgToAnns.keys())
        clsid2catid = cocoGt.getCatIds()
        with open(os.path.join(map_out_root, 'eval_results.json'), "w") as f:
            results = []
            for image_id in tqdm(ids):
                image_path = os.path.join(dataset_img_path, cocoGt.loadImgs(image_id)[0]['file_name'])

                # 图片预处理
                image, h, w = read_image_and_convert_to_tensor(image_path, size=self.cfg.arch.input_size[1:],
                                                               letterbox=self.letterbox_image)
                image = image.to(self.device)

                with torch.no_grad():
                    preds = model(image)
                    top_boxes, top_conf, top_label = self.decode_boxes(preds, h, w, conf_threshold=0.001)

                if top_boxes.shape[0] == 0:
                    continue

                for i, c in enumerate(top_label):
                    result = {}
                    left, top, right, bottom = top_boxes[i]

                    result["image_id"] = int(image_id)
                    result["category_id"] = clsid2catid[c]
                    result["bbox"] = [float(left), float(top), float(right - left), float(bottom - top)]
                    result["score"] = float(top_conf[i])
                    results.append(result)

            json.dump(results, f)

        cocoDt = cocoGt.loadRes(os.path.join(map_out_root, 'eval_results.json'))
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        print("Get map done.")

    def decode_boxes(self, pred, h, w, conf_threshold=None):
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        heatmap = pred[..., :self.num_classes]
        reg = pred[..., self.num_classes: self.num_classes + 2]
        wh = pred[..., -2:]
        batch_size = heatmap.size(0)

        heatmap = torch.sigmoid(heatmap)
        heatmap = CenterNetA._suppress_redundant_centers(heatmap)
        scores, inds, classes, ys, xs = CenterNetA._top_k(scores=heatmap, k=self.K)
        if reg is not None:
            reg = RegL1Loss.gather_feat(feat=reg, ind=inds)
            xs = torch.reshape(xs, shape=(batch_size, self.K)) + reg[:, :, 0]  # shape: (batch_size, self.K)
            ys = torch.reshape(ys, shape=(batch_size, self.K)) + reg[:, :, 1]
        else:
            xs = torch.reshape(xs, shape=(batch_size, self.K)) + 0.5
            ys = torch.reshape(ys, shape=(batch_size, self.K)) + 0.5
        wh = RegL1Loss.gather_feat(feat=wh, ind=inds)  # shape: (batch_size, self.K, 2)
        classes = torch.reshape(classes, (batch_size, self.K))
        scores = torch.reshape(scores, (batch_size, self.K))

        bboxes = torch.cat(tensors=[xs.unsqueeze(-1), ys.unsqueeze(-1), wh], dim=-1)  # shape: (batch_size, self.K, 4)
        bboxes[..., ::2] /= self.feature_size[1]
        bboxes[..., 1::2] /= self.feature_size[0]
        bboxes = torch.clamp(bboxes, min=0, max=1)
        # (cx, cy, w, h) ----> (xmin, ymin, xmax, ymax)
        bboxes = xywh_to_xyxy_torch(bboxes)

        score_mask = scores >= conf_threshold  # shape: (batch_size, self.K)

        bboxes = bboxes[score_mask]
        scores = scores[score_mask]
        classes = classes[score_mask]
        if self.use_nms:
            indices = diou_nms(boxes=bboxes, scores=scores, iou_threshold=self.nms_threshold)
            bboxes, scores, classes = bboxes[indices], scores[indices], classes[indices]

        boxes = reverse_letter_box(h=h, w=w, input_size=self.input_size, boxes=bboxes, xywh=False)
        # 转化为numpy.ndarray
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        classes = classes.cpu().numpy()
        return boxes, scores, classes

    @staticmethod
    def _suppress_redundant_centers(heatmap, pool_size=3):
        """
        消除8邻域内的其它峰值点
        :param heatmap:
        :param pool_size:
        :return:
        """
        hmax = torch.nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=((pool_size - 1) // 2))(heatmap)
        keep = torch.eq(heatmap, hmax).to(torch.float32)
        return heatmap * keep

    @staticmethod
    def _top_k(scores, k):
        B, H, W, C = scores.size()
        scores = torch.reshape(scores, shape=(B, -1))
        topk_scores, topk_inds = torch.topk(input=scores, k=k, largest=True, sorted=True)
        topk_clses = topk_inds % C  # 应该选取哪些通道（类别）
        pixel = torch.div(topk_inds, C, rounding_mode="floor")
        topk_ys = torch.div(pixel, W, rounding_mode="floor")  # 中心点的y坐标
        topk_xs = pixel % W  # 中心点的x坐标
        topk_inds = (topk_ys * W + topk_xs).to(torch.int32)
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
