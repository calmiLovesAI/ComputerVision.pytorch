import json
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from configs.dataset_cfg import COCO_CFG, VOC_CFG
import xml.etree.ElementTree as ET

from core.loss.ultralytics_loss import BboxLoss
from core.metrics.mAP import get_coco_map, get_map
from core.models.yolov8.yolo_v8 import get_yolo8_n, get_yolo8_s, get_yolo8_m, get_yolo8_l, get_yolo8_x
from core.utils.anchor import make_anchors
from core.utils.bboxes import TaskAlignedAssigner, dist2bbox
from core.utils.image_process import read_image_and_convert_to_tensor, read_image, yolo_correct_boxes
from core.utils.ultralytics_ops import non_max_suppression, xywh2xyxy
from core.utils.visualize import show_detection_results
from registry import model_registry
from configs import Yolo8DetConfig

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class Loss:

    def __init__(self, cfg, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        # h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        self.box = cfg.loss.box
        self.cls = cfg.loss.cls
        self.dfl = cfg.loss.dfl

        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        # (bs, 16*4, 8400), (bs, nc, 8400)
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (bs, 8400, nc)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (bs, 8400, 16*4)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        # anchor_points: torch.Size([8400, 2]), stride_tensor: torch.Size([8400, 1])
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets shape: [N, 6]
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        # (bs, num_max_true_boxes, 5)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])

        # (bs, num_max_true_boxes, 1), (bs, num_max_true_boxes, 4)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)   # (bs, num_max_true_boxes, 1)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, 8400, 4)

        # target_bboxes: [bs, 8400, 4], target_scores: [bs, 8400, 80]
        # fg_mask: torch.Size([bs, 8400])
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.box  # box gain
        loss[1] *= self.cls  # cls gain
        loss[2] *= self.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


@model_registry("yolo8_det")
class YOLOv8:
    def __init__(self, cfg: Yolo8DetConfig, device):
        self.cfg = cfg
        self.device = device

        self.model_type = self.cfg.arch.model_type
        # 类别数目
        self.num_classes = self.cfg.dataset.num_classes
        # 输入图片的尺寸
        self.input_image_size = self.cfg.arch.input_size[1:]

        # 解码
        self.conf_threshold = self.cfg.decode.conf_threshold
        self.iou_threshold = self.cfg.decode.nms_threshold
        self.max_det = self.cfg.decode.max_det
        self.letterbox_image = self.cfg.decode.letterbox_image

    def build_model(self):
        """
        构建网络模型
        :return:
        """
        if self.model_type == "n":
            model = get_yolo8_n(nc=self.num_classes)
            model_name = "YOLOv8n"
        elif self.model_type == "s":
            model = get_yolo8_s(nc=self.num_classes)
            model_name = "YOLOv8s"
        elif self.model_type == "m":
            model = get_yolo8_m(nc=self.num_classes)
            model_name = "YOLOv8m"
        elif self.model_type == "l":
            model = get_yolo8_l(nc=self.num_classes)
            model_name = "YOLOv8l"
        elif self.model_type == "x":
            model = get_yolo8_x(nc=self.num_classes)
            model_name = "YOLOv8x"
        else:
            raise ValueError(f"model_type: {self.model_type} is not supported")
        return model, model_name

    def build_loss(self, model):
        return Loss(cfg=self.cfg, model=model)

    def predict(self, model, image_path, print_on, save_result):
        """
        模型预测
        :param model:
        :param image_path:
        :param print_on:
        :param save_result:
        :return:
        """
        model.eval()
        # 处理单张图片
        image, h, w = read_image_and_convert_to_tensor(
            image_path, size=self.cfg.arch.input_size[1:], letterbox=self.letterbox_image)
        image = image.to(self.device)

        with torch.no_grad():
            # 图片输入到模型中，得到预测输出
            preds = model(image)
            # 解码
            results = self.decode_box(preds, h, w)

        if results[0].shape[0] == 0:
            print(f"No object detected")
            return read_image(image_path, mode='bgr')

        # 得到更详细的边界框信息、分数信息和类别信息
        boxes, scores, classes = results[0], results[1], results[2]

        # 将检测结果绘制在原始图片上
        return show_detection_results(image_path=image_path,
                                      dataset_name=self.cfg.dataset.dataset_name,
                                      boxes=boxes,
                                      scores=scores,
                                      class_indices=classes,
                                      print_on=print_on,
                                      save_result=save_result,
                                      save_dir=self.cfg.decode.test_results)

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

        preds = non_max_suppression(preds,
                                    conf_threshold,
                                    self.iou_threshold,
                                    agnostic=False,
                                    max_det=self.max_det,
                                    classes=None)

        assert len(preds) == 1, "仅支持单张图片的预测"
        pred = preds[0].cpu().numpy()
        bbox, conf, cls = pred[:, :4], pred[:, 4], pred[:, 5].astype(np.int)
        # 坐标归一化到0~1范围内
        bbox[:, ::2] /= self.input_image_size[1]
        bbox[:, 1::2] /= self.input_image_size[0]

        # 计算相对于原始输入图片的边界框坐标
        box_xy, box_wh = (bbox[:, 0:2] + bbox[:, 2:4]
                          ) / 2, bbox[:, 2:4] - bbox[:, 0:2]
        bbox[:, :4] = yolo_correct_boxes(box_xy, box_wh,
                                         self.input_image_size, [image_h, image_w],
                                         self.letterbox_image)
        return bbox, conf, cls
    

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

                if results[0].shape[0] == 0:
                    results[0] = np.zeros(shape=(1, 4), dtype=np.float32)
                
                boxes = results[0]
                scores = results[1]
                class_indices = results[2]

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

                if outputs[0].shape[0] == 0:
                    outputs[0] = np.zeros(shape=(1, 4), dtype=np.float32)
                
                top_boxes = outputs[0]
                top_conf = outputs[1]
                top_label = outputs[2]

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
