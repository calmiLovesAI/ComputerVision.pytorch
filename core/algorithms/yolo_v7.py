import json
import os

import numpy as np
import torch
import xml.etree.ElementTree as ET
from tqdm import tqdm

from configs.dataset_cfg import VOC_CFG, COCO_CFG
# from configs.yolo7_cfg import Config
from core.loss.yolo7_loss import Yolo7Loss
from core.metrics.mAP import get_map, get_coco_map
from core.models.yolov7_model import Yolo7
from core.utils.bboxes import xywh_to_xyxy_torch
from core.utils.image_process import read_image_and_convert_to_tensor, yolo_correct_boxes, read_image
from core.utils.visualize import show_detection_results
from torchvision.ops import nms

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from registry import model_registry
from configs import Yolo7Config


@model_registry("yolo7")
class YOLOv7:

    def __init__(self, cfg: Yolo7Config, device) -> None:
        self.cfg = cfg
        self.device = device
        # 锚框
        self.anchors = self.get_anchors()
        # 类别数目
        self.num_classes = self.cfg.dataset.num_classes
        # 输入图片的尺寸
        self.input_image_size = self.cfg.arch.input_size[1:]
        # 每个bounding box的属性数目，5代表(x, y, w, h, obj)
        self.bbox_attrs = 5 + self.num_classes
        self.anchors_mask = self.cfg.arch.anchors_mask
        self.letterbox_image = self.cfg.decode.letterbox_image
        self.conf_threshold = self.cfg.decode.conf_threshold
        self.nms_threshold = self.cfg.decode.nms_threshold

    def get_anchors(self) -> np.ndarray:
        """获取anchors"""
        anchors_list = self.cfg.arch.anchors
        anchors = np.array(anchors_list, dtype=np.float32).reshape(-1, 2)
        return anchors

    def build_model(self):
        """构建网络模型"""
        model = Yolo7(self.cfg)
        model_name = "YOLOv7"
        return model, model_name

    def build_loss(self):
        """构建损失函数"""
        loss = Yolo7Loss(anchors=self.anchors,
                         num_classes=self.num_classes,
                         input_shape=self.input_image_size,
                         anchors_mask=self.cfg.arch.anchors_mask,
                         label_smoothing=self.cfg.loss.label_smoothing)
        return loss

    def predict(self, model, image_path, print_on, save_result):
        model.eval()
        # 处理单张图片
        image, h, w = read_image_and_convert_to_tensor(
            image_path, size=self.cfg.arch.input_size[1:], letterbox=self.letterbox_image)
        image = image.to(self.device)

        with torch.no_grad():
            preds = model(image)
            results = self.decode_box(preds, h, w)

        if results[0] is None:
            print(f"No object detected")
            return read_image(image_path, mode='bgr')

        boxes = results[0][:, :4]
        scores = results[0][:, 4] * results[0][:, 5]
        classes = results[0][:, 6].astype(np.int32)

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
                    results = self.decode_box(preds, h, w, conf_threshold=0.001)

                if results[0] is None:
                    # 填充0
                    results[0] = np.zeros(shape=(1, 7), dtype=np.float32)

                boxes = results[0][:, :4]
                scores = results[0][:, 4] * results[0][:, 5]
                class_indices = results[0][:, 6].astype(np.int32)

                # 将检测结果写入txt文件中
                with open(file=os.path.join(detections_path, f"{image_id}.txt"), mode='w', encoding='utf-8') as f:
                    for i, c in enumerate(class_indices):
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
                    outputs = self.decode_box(preds, h, w, conf_threshold=0.001)

                if outputs[0] is None:
                    continue

                top_boxes = outputs[0][:, :4]
                top_conf = outputs[0][:, 4] * outputs[0][:, 5]
                top_label = outputs[0][:, 6].astype('int32')

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

    def decode_box(self, preds, image_h, image_w, conf_threshold=None):
        """
        解码预测结果
        :param preds: 预测结果
        :param image_h: 图片高度
        :param image_w: 图片宽度
        :param conf_threshold: 置信度
        :return: 解码后的预测结果
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        outputs = []
        for i, pred in enumerate(preds):
            # -----------------------------------------------#
            #   输入的pred一共有三个，他们的shape分别是
            #   batch_size = 1
            #   batch_size, 3 * (4 + 1 + 80), 20, 20
            #   batch_size, 255, 40, 40
            #   batch_size, 255, 80, 80
            # -----------------------------------------------#
            bs, _, input_height, input_width = pred.size()
            stride_h = self.input_image_size[0] / input_height
            stride_w = self.input_image_size[1] / input_width

            # 将anchors的尺寸根据其对应的feature的缩放比例进行缩放
            scaled_anchors = [(anchor_width / stride_w,
                               anchor_height / stride_h)
                              for anchor_width, anchor_height in self.anchors[
                                  self.anchors_mask[i]]]
            # -----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 3, 20, 20, 85
            #   batch_size, 3, 40, 40, 85
            #   batch_size, 3, 80, 80, 85
            # -----------------------------------------------#
            pred = torch.reshape(pred,
                                 shape=(bs, len(self.anchors_mask[i]),
                                        self.bbox_attrs, input_height,
                                        input_width)).permute(0, 1, 3, 4, 2)
            # 先验框的中心位置的调整参数
            x = torch.sigmoid(pred[..., 0])
            y = torch.sigmoid(pred[..., 1])
            # 先验框的宽高调整参数
            w = torch.sigmoid(pred[..., 2])
            h = torch.sigmoid(pred[..., 3])
            # 先验框的置信度
            pred_conf = torch.sigmoid(pred[..., 4])
            # 先验框的种类置信度
            pred_cls = torch.sigmoid(pred[..., 5:])

            # 生成网格
            grid_x = torch.linspace(
                start=0, end=input_width - 1,
                steps=input_width).repeat(input_height, 1).repeat(
                bs * len(self.anchors_mask[i]), 1,
                1).view(x.shape).to(torch.float32).to(self.device)
            grid_y = torch.linspace(
                start=0, end=input_height - 1,
                steps=input_height).repeat(input_width, 1).t().repeat(
                bs * len(self.anchors_mask[i]), 1,
                1).view(y.shape).to(torch.float32).to(self.device)

            # ----------------------------------------------------------#
            #   按照网格格式生成先验框的宽高
            #   batch_size, 3, 20, 20
            # ----------------------------------------------------------#
            anchor_w = torch.tensor(scaled_anchors,
                                    dtype=torch.float32,
                                    device=self.device).index_select(
                1,
                torch.tensor([0],
                             dtype=torch.int64,
                             device=self.device))
            anchor_w = anchor_w.repeat(bs, 1).repeat(
                1, 1, input_height * input_width).view(w.shape)

            anchor_h = torch.tensor(scaled_anchors,
                                    dtype=torch.float32,
                                    device=self.device).index_select(
                1,
                torch.tensor([1],
                             dtype=torch.int64,
                             device=self.device))
            anchor_h = anchor_h.repeat(bs, 1).repeat(
                1, 1, input_height * input_width).view(h.shape)

            # ----------------------------------------------------------#
            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            #   x 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   y 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   w 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            #   h 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            # ----------------------------------------------------------#
            pred_boxes = torch.zeros_like(pred[..., :4])
            pred_boxes[..., 0] = x.detach() * 2. - 0.5 + grid_x
            pred_boxes[..., 1] = y.detach() * 2. - 0.5 + grid_y
            pred_boxes[..., 2] = (w.detach() * 2) ** 2 * anchor_w
            pred_boxes[..., 3] = (h.detach() * 2) ** 2 * anchor_h
            # 将输出结果归一化成小数的形式
            _scale = torch.tensor(
                [input_width, input_height, input_width, input_height],
                dtype=torch.float32,
                device=self.device)
            output = torch.cat((pred_boxes.reshape(bs, -1, 4) / _scale,
                                pred_conf.reshape(bs, -1, 1),
                                pred_cls.reshape(bs, -1, self.num_classes)),
                               -1)
            outputs.append(output.detach())
        decoded_outputs = torch.cat(outputs, 1)
        results = self._nms(decoded_outputs, self.input_image_size, [image_h, image_w], conf_threshold)
        return results

    def _nms(self, prediction, input_shape, image_shape, conf_threshold):
        """
        非极大抑制
        :param prediction: 预测结果
        :param input_shape: 网络输入图片的shape
        :param image_shape: 测试图片的shape
        :param conf_threshold:
        :return: 经过非极大抑制后的结果
        """
        # ----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        # ----------------------------------------------------------#
        prediction = xywh_to_xyxy_torch(prediction, more=True)

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # ----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            # ----------------------------------------------------------#
            class_conf, class_pred = torch.max(image_pred[:, 5:5 +
                                                               self.num_classes],
                                               1,
                                               keepdim=True)
            # ----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            # ----------------------------------------------------------#
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >=
                         conf_threshold).squeeze()
            # ----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            # ----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue
            # -------------------------------------------------------------------------#
            #   detections  [num_anchors, 7]
            #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
            # -------------------------------------------------------------------------#
            detections = torch.cat(
                (image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
            # ------------------------------------------#
            #   获得预测结果中包含的所有种类
            # ------------------------------------------#
            unique_labels = detections[:, -1].unique()

            for c in unique_labels:
                # ------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                # ------------------------------------------#
                detections_class = detections[detections[:, -1] == c]
                # ------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                #   筛选出一定区域内，属于同一种类得分最大的框
                # ------------------------------------------#
                keep = nms(boxes=detections_class[:, :4],
                           scores=detections_class[:, 4] *
                                  detections_class[:, 5],
                           iou_threshold=self.nms_threshold)
                max_detections = detections_class[keep]
                output[i] = max_detections if output[i] is None else torch.cat(
                    (output[i], max_detections))

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]
                                  ) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = yolo_correct_boxes(box_xy, box_wh,
                                                      input_shape, image_shape,
                                                      self.letterbox_image)
        return output
