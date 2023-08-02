import json
import os

import numpy as np
import torch
import xml.etree.ElementTree as ET

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import nms
from tqdm import tqdm

from configs.dataset_cfg import VOC_CFG, COCO_CFG
from core.loss.multi_box_loss import MultiBoxLossV2
from core.metrics.mAP import get_map, get_coco_map
from core.models.ssd_model import SSD
from core.utils.bboxes import xywh_to_xyxy
from core.utils.image_process import read_image_and_convert_to_tensor, read_image, yolo_correct_boxes
from core.utils.visualize import show_detection_results

from registry import model_registry
from configs import SsdConfig


@model_registry("ssd")
class Ssd:

    def __init__(self, cfg: SsdConfig, device):
        self.cfg = cfg
        self.device = device
        # 输入图片的尺寸
        self.input_image_size = self.cfg.arch.input_size[1:]
        # 与锚框有关的参数
        self.anchor_sizes = self.cfg.arch.anchor_sizes
        self.feature_shapes = self.cfg.arch.feature_shapes
        self.aspect_ratios = self.cfg.arch.aspect_ratios
        # 锚框
        self.anchors = self._get_ssd_anchors()
        self.num_anchors = self.anchors.shape[0]
        # 类别数目
        self.num_classes = self.cfg.dataset.num_classes
        # 正负样本比例
        self.neg_pos_ratio = self.cfg.loss.neg_pos
        variance = np.array(self.cfg.loss.variance, dtype=np.float32)
        # 将variance变成[0.1, 0.1, 0.2, 0.2]
        self.variance = np.repeat(variance, 2, axis=0)
        self.overlap_threshold = self.cfg.loss.overlap_threshold

        self.conf_threshold = self.cfg.decode.confidence_threshold
        self.nms_threshold = self.cfg.decode.nms_threshold
        self.letterbox_image = self.cfg.decode.letterbox_image

    def build_model(self):
        """构建网络模型"""
        if self.cfg.arch.backbone == "vgg":
            return SSD(self.cfg), f"SSD{self.input_image_size[0]}_vgg"
        elif self.cfg.arch.backbone == "mobilenetv2":
            # TODO MobileNetV2作为骨干网络
            pass


    def build_loss(self):
        """构建损失函数"""
        loss = MultiBoxLossV2(neg_pos_ratio=self.neg_pos_ratio,
                              num_classes=self.num_classes)
        return loss

    def predict(self, model, image_path, print_on, save_result):
        model.eval()
        # 处理单张图片
        image, h, w = read_image_and_convert_to_tensor(
            image_path, size=self.cfg.arch.input_size[1:], letterbox=self.letterbox_image)
        image = image.to(self.device)

        with torch.no_grad():
            preds = model(image)
            results = self.decode_boxes(preds, h, w)

        if len(results[0]) == 0:
            print(f"No object detected")
            return read_image(image_path, mode='bgr')

        boxes = results[0][:, :4]
        scores = results[0][:, 5]
        classes = results[0][:, 4].astype(np.int32)

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
                    results = self.decode_boxes(preds, h, w, conf_threshold=0.001)

                if len(results[0]) == 0:
                    # 填充0
                    results[0] = np.zeros(shape=(1, 6), dtype=np.float32)

                boxes = results[0][:, :4]
                scores = results[0][:, 5]
                classes = results[0][:, 4].astype(np.int32)

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
                    outputs = self.decode_boxes(preds, h, w, conf_threshold=0.001)

                if len(outputs[0]) == 0:
                    continue

                top_boxes = outputs[0][:, :4]
                top_conf = outputs[0][:, 5]
                top_label = outputs[0][:, 4].astype('int32')

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

    def decode_boxes(self, preds, h, w, conf_threshold=None):
        """
        :param preds: SSD网络模型的输出 tuple: (loc, conf) 其中loc.shape = (batch, 8732, 4), conf.shape = (batch, 8732, num_classes + 1)
        :param h:  输入图片的高
        :param w:  输入图片的宽
        :return:
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        # 回归预测结果，(batch, 8732, 4)
        mbox_loc = preds[0]
        # 置信度预测结果，(batch, 8732, num_classes + 1)
        mbox_conf = torch.softmax(preds[1], dim=-1)
        batch_size = mbox_loc.size(0)

        results = []
        for i in range(batch_size):
            results.append([])
            # 解析mbox_loc，得到真正的回归坐标
            decode_bbox = self._parse_mbox_loc(mbox_loc[i])
            for c in range(1, self.num_classes + 1):
                # 属于该类的所有框的置信度
                c_confs = mbox_conf[i, :, c]
                # 判断是否大于门限
                c_confs_m = c_confs > conf_threshold
                if len(c_confs[c_confs_m]) > 0:
                    # 取出得分高于confidence的框
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]

                    # 非极大值抑制
                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        self.nms_threshold
                    )
                    # 取出在非极大抑制中效果较好的内容
                    good_boxes = boxes_to_process[keep]
                    confs = confs_to_process[keep][:, None]
                    labels = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones(
                        (len(keep), 1))
                    # 将框的位置、label、置信度进行堆叠。
                    c_pred = torch.cat((good_boxes, labels, confs), dim=1).cpu().numpy()
                    # 添加进result里
                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4]) / 2, results[-1][:, 2:4] - results[-1][:,
                                                                                                        0:2]
                results[-1][:, :4] = yolo_correct_boxes(box_xy, box_wh, self.input_image_size, [h, w],
                                                        self.letterbox_image)
        return results

    def _parse_mbox_loc(self, mbox_loc):
        variances = self.variance[::2].tolist()
        anchors = torch.from_numpy(self.anchors).to(torch.float32).to(self.device)
        # 获得先验框的宽与高
        anchor_width = anchors[:, 2] - anchors[:, 0]
        anchor_height = anchors[:, 3] - anchors[:, 1]
        # 获得先验框的中心点
        anchor_center_x = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchor_center_y = 0.5 * (anchors[:, 3] + anchors[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = mbox_loc[:, 0] * anchor_width * variances[0]
        decode_bbox_center_x += anchor_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * anchor_height * variances[0]
        decode_bbox_center_y += anchor_center_y

        # 真实框的宽与高
        decode_bbox_width = torch.exp(mbox_loc[:, 2] * variances[1])
        decode_bbox_width *= anchor_width
        decode_bbox_height = torch.exp(mbox_loc[:, 3] * variances[1])
        decode_bbox_height *= anchor_height

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = torch.cat((decode_bbox_xmin[:, None],
                                 decode_bbox_ymin[:, None],
                                 decode_bbox_xmax[:, None],
                                 decode_bbox_ymax[:, None]), dim=-1)
        # 防止超出0与1
        decode_bbox = torch.min(torch.max(decode_bbox, torch.zeros_like(decode_bbox)), torch.ones_like(decode_bbox))
        return decode_bbox

    def generate_targets(self, label):
        """
        将一个Nx6的Tensor变成一个8732x5的Tensor
        :param label: numpy.ndarray, shape: (N, 6(_, class_id, cx, cy, w, h))
        :return: torch.Tensor, shape: (8732, 5(center_x, center_y, w, h, label))
        """
        # 数据集的类别标签增加1，因为背景的类别是0
        label[:, 1] += 1
        class_label = label[:, 1].astype(np.int32)
        # 坐标 (N, 4)
        coord_label = label[:, 2:]
        # 坐标由(cx, cy, w, h)转换为(xmin, ymin, xmax, ymax)
        coord_label = xywh_to_xyxy(coord_label)
        # one-hot编码  (N, 21)
        one_hot_label = np.eye(self.num_classes + 1)[class_label]
        # 包含坐标和one-hot编码的标签的label (N, 4 + 21)
        true_label = np.concatenate((coord_label, one_hot_label), axis=-1)

        # assignment[:, :4] 坐标
        # assignment[:, 4:-1] one-hot编码
        # assignment[:, -1] 当前先验框是否有对应的目标，0为没有，1为有
        assignment = np.zeros((self.num_anchors, 4 + 1 + self.num_classes + 1),
                              dtype=np.float32)
        assignment[:, 4] = 1.0  # 默认先验框为背景
        if len(true_label) == 0:
            return torch.from_numpy(assignment)
        # 对每一个真实框都进行iou计算
        encoded_boxes = np.apply_along_axis(self._encode_box, 1,
                                            true_label[:, :4])

        # ---------------------------------------------------#
        #   在reshape后，获得的encoded_boxes的shape为：
        #   [num_true_box, num_anchors, 4 + 1]
        #   4是编码后的结果，1为iou
        # ---------------------------------------------------#
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)

        # ---------------------------------------------------#
        #   [num_anchors]求取每一个先验框重合度最大的真实框
        # ---------------------------------------------------#
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        # ---------------------------------------------------#
        #   计算一共有多少先验框满足需求
        # ---------------------------------------------------#
        assign_num = len(best_iou_idx)

        # 将编码后的真实框取出
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        # ---------------------------------------------------#
        #   编码后的真实框的赋值
        # ---------------------------------------------------#
        assignment[:, :4][best_iou_mask] = encoded_boxes[
                                           best_iou_idx, np.arange(assign_num), :4]
        # ----------------------------------------------------------#
        #   4代表为背景的概率，设为0，因为这些先验框有对应的物体
        # ----------------------------------------------------------#
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-1][best_iou_mask] = true_label[best_iou_idx, 5:]
        # ----------------------------------------------------------#
        #   -1表示先验框是否有对应的物体
        # ----------------------------------------------------------#
        assignment[:, -1][best_iou_mask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return torch.from_numpy(assignment)

    def _encode_box(self, box, return_iou=True):
        # ---------------------------------------------#
        #   计算当前真实框和先验框的重合情况
        #   iou [self.num_anchors]
        #   encoded_box [self.num_anchors, 5]
        # ---------------------------------------------#

        # ---------------------------------------------#
        #   计算出每个真实框与所有的先验框的iou
        #   判断真实框与先验框的重合情况
        # ---------------------------------------------#
        inter_upleft = np.maximum(self.anchors[:, :2], box[:2])
        inter_botright = np.minimum(self.anchors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # ---------------------------------------------#
        #   真实框的面积
        # ---------------------------------------------#
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # ---------------------------------------------#
        #   先验框的面积
        # ---------------------------------------------#
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0]) * (
                self.anchors[:, 3] - self.anchors[:, 1])
        # ---------------------------------------------#
        #   计算iou
        # ---------------------------------------------#
        union = area_true + area_gt - inter

        iou = inter / union

        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))

        # ---------------------------------------------#
        #   找到每一个真实框，重合程度较高的先验框
        #   真实框可以由这个先验框来负责预测
        # ---------------------------------------------#
        assign_mask = iou > self.overlap_threshold

        # ---------------------------------------------#
        #   如果没有一个先验框重合度大于self.overlap_threshold
        #   则选择重合度最大的为正样本
        # ---------------------------------------------#
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        # ---------------------------------------------#
        #   利用iou进行赋值
        # ---------------------------------------------#
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        # ---------------------------------------------#
        #   找到对应的先验框
        # ---------------------------------------------#
        assigned_anchors = self.anchors[assign_mask]

        # ---------------------------------------------#
        #   逆向编码，将真实框转化为ssd预测结果的格式
        #   先计算真实框的中心与长宽
        # ---------------------------------------------#
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        # ---------------------------------------------#
        #   再计算重合度较高的先验框的中心与长宽
        # ---------------------------------------------#
        assigned_anchors_center = (assigned_anchors[:, 0:2] +
                                   assigned_anchors[:, 2:4]) * 0.5
        assigned_anchors_wh = (assigned_anchors[:, 2:4] -
                               assigned_anchors[:, 0:2])

        # ------------------------------------------------#
        #   逆向求取ssd应该有的预测结果
        #   先求取中心的预测结果，再求取宽高的预测结果
        #   存在改变数量级的参数，默认为[0.1,0.1,0.2,0.2]
        # ------------------------------------------------#
        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(self.variance)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(self.variance)[2:4]
        return encoded_box.ravel()

    def _get_ssd_anchors(self):
        image_h, image_w = self.input_image_size
        anchors = []
        for i in range(len(self.feature_shapes)):
            # 先验框的短边和长边
            min_size = self.anchor_sizes[i]
            max_size = self.anchor_sizes[i + 1]
            # 特征图的高和宽，它们相等
            feature_h = self.feature_shapes[i]
            # 对于每个像素位置，根据aspect_ratio生成不同宽、高比的先验框
            box_widths = []
            box_heights = []
            for ar in self.aspect_ratios[i]:
                if ar == 1:
                    box_widths.append(min_size)
                    box_heights.append(min_size)
                    box_widths.append(np.sqrt(min_size * max_size))
                    box_heights.append(np.sqrt(min_size * max_size))
                else:
                    box_widths.append(min_size * np.sqrt(ar))
                    box_heights.append(min_size / np.sqrt(ar))

            half_box_widths = np.array(
                box_widths) / 2.0  # shape: (len(aspect_ratios[i])+1,)
            half_box_heights = np.array(box_heights) / 2.0

            # 特征层上一个像素点映射到原图上对应的像素长度
            pixel_length = [image_h / feature_h, image_w / feature_h]
            # 生成网格中心
            c_x = np.linspace(0.5 * pixel_length[1],
                              image_w - 0.5 * pixel_length[1], feature_h)
            c_y = np.linspace(0.5 * pixel_length[0],
                              image_h - 0.5 * pixel_length[0], feature_h)
            center_x, center_y = np.meshgrid(c_x, c_y)
            center_x = np.reshape(center_x, (-1, 1))  # (feature_h**2, 1)
            center_y = np.reshape(center_y, (-1, 1))  # (feature_h**2, 1)

            anchor = np.concatenate((center_x, center_y),
                                    axis=1)  # (feature_h**2, 2)
            # 对于每一种宽高比例，都需要一个对应的先验框
            # shape: (feature_h**2, 4*(len(aspect_ratios[i])+1))
            anchor = np.tile(anchor, (1, (len(self.aspect_ratios[i]) + 1) * 2))

            # 转换为xmin, ymin, xmax, ymax格式
            # shape: (feature_h**2, len(aspect_ratios[i])+1)
            anchor[:, ::4] -= half_box_widths
            anchor[:, 1::4] -= half_box_heights
            anchor[:, 2::4] += half_box_widths
            anchor[:, 3::4] += half_box_heights

            # 归一化
            anchor[:, ::2] /= image_w
            anchor[:, 1::2] /= image_h
            anchor = np.clip(anchor, a_min=0.0, a_max=1.0)
            anchor = np.reshape(anchor, (-1, 4))

            anchors.append(anchor)

        anchors = np.concatenate(anchors, axis=0)  # (8732, 4)
        return anchors.astype(dtype=np.float32)
