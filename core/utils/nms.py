import torch
from torchvision.ops import nms

from core.utils.bboxes import xywh_to_xyxy_torch
from core.utils.image_process import yolo_correct_boxes
from core.utils.iou import box_diou


def diou_nms(boxes, scores, iou_threshold):
    """

    :param boxes: (Tensor[N, 4]) – boxes to perform NMS on. They are expected to be in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.
    :param scores: (Tensor[N]) – scores for each one of the boxes
    :param iou_threshold: (float) – discards all overlapping boxes with DIoU > iou_threshold
    :return: int64 tensor with the indices of the elements that have been kept by DIoU-NMS, sorted in decreasing order of scores
    """
    order = torch.argsort(scores, dim=0, descending=True)
    keep = list()
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        else:
            index = order[0]
            keep.append(index)
        value = box_diou(boxes1=boxes[index], boxes2=boxes[order[1:]])
        mask_index = (value <= iou_threshold).nonzero().squeeze()
        if mask_index.numel() == 0:
            break
        order = order[mask_index + 1]
    return torch.LongTensor(keep)


def gather_op(tensor, indice, device):
    """

    :param tensor: shape: (M, N)
    :param indice: shape: (K,)
    :return: Tensor, shape: (K, N)
    """
    assert tensor.dim() == 1 or tensor.dim() == 2
    if tensor.dim() == 2:
        M, N = tensor.size()
    if tensor.dim() == 1:
        M = tensor.size()[0]
        N = 1
    K = indice.size()[0]
    container = torch.zeros(K, N, dtype=torch.float32, device=device)
    for k in range(K):
        container[k] = tensor[indice[k]]
    return container


def yolo3_nms(num_classes,
              conf_threshold,
              iou_threshold,
              boxes,
              scores,
              device):
    mask = scores >= conf_threshold

    box_list = list()
    score_list = list()
    class_list = list()

    for i in range(num_classes):
        box_of_class = boxes[mask[:, i]]
        score_of_class = scores[mask[:, i], i]
        indices = nms(boxes=box_of_class, scores=score_of_class, iou_threshold=iou_threshold)
        selected_boxes = gather_op(box_of_class, indices, device)
        selected_scores = gather_op(score_of_class, indices, device)
        select_classes = torch.ones(*selected_scores.size(), dtype=torch.int32, device=device) * i

        box_list.append(selected_boxes)
        score_list.append(selected_scores)
        class_list.append(select_classes)

    boxes = torch.cat(box_list, dim=0)
    scores = torch.cat(score_list, dim=0)
    classes = torch.cat(class_list, dim=0)

    classes = torch.squeeze(classes, dim=1)

    return boxes, scores, classes


def yolo7_nms(prediction, num_classes, input_shape, image_shape, letterbox_image, device, conf_thres=0.5,
              nms_thres=0.4):
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
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        # ----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        # ----------------------------------------------------------#
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
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
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
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
            keep = nms(
                boxes=detections_class[:, :4],
                scores=detections_class[:, 4] * detections_class[:, 5],
                iou_threshold=nms_thres
            )
            max_detections = detections_class[keep]
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4] = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return output
