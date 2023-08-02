import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from core.utils.ultralytics_iou import bbox_iou


def xywh_to_xyxy(coords):
    """
    坐标变换
    :param coords: numpy.ndarray, 最后一维的4个数是坐标 (center_x, center_y, w, h)
    :return: numpy.ndarray, 与输入的形状一致，最后一维的格式是(xmin, ymin, xmax, ymax)
    """
    cx = coords[..., 0:1]
    cy = coords[..., 1:2]
    w = coords[..., 2:3]
    h = coords[..., 3:4]

    xmin = cx - w / 2
    xmax = cx + w / 2
    ymin = cy - h / 2
    ymax = cy + h / 2

    new_coords = np.concatenate((xmin, ymin, xmax, ymax), axis=-1)
    return new_coords


def xywh_to_xyxy_torch(coords, more=False):
    """
    坐标变换
    :param coords: torch.Tensor, 最后一维的4个数是坐标 (center_x, center_y, w, h)
    :param more: 最后一维除了前4个数是坐标外，还有更多的数
    :return: torch.Tensor, 与输入的形状一致，最后一维的格式是(xmin, ymin, xmax, ymax)
    """
    cx = coords[..., 0:1]
    cy = coords[..., 1:2]
    w = coords[..., 2:3]
    h = coords[..., 3:4]

    xmin = cx - w / 2
    xmax = cx + w / 2
    ymin = cy - h / 2
    ymax = cy + h / 2

    new_coords = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
    if more:
        new_coords = torch.cat([new_coords, coords[..., 4:]], dim=-1)
    return new_coords


def xyxy_to_xywh(coords, center=True):
    """
    坐标变换
    :param coords: numpy.ndarray, 最后一维的4个数是坐标
    :param center: True表示将(xmin, ymin, xmax, ymax)转变为(center_x, center_y, w, h)格式
                   False表示将(xmin, ymin, xmax, ymax)转变为(xmin, ymin, w, h)格式
    :return: numpy.ndarray, 与输入的形状一致
    """
    xmin = coords[..., 0:1]
    ymin = coords[..., 1:2]
    xmax = coords[..., 2:3]
    ymax = coords[..., 3:4]
    w = xmax - xmin
    h = ymax - ymin
    if center:
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        return np.concatenate((center_x, center_y, w, h), axis=-1)
    else:
        return np.concatenate((xmin, ymin, w, h), axis=-1)


def xyxy_to_xywh_torch(coords, center=True):
    """
    坐标变换
    :param coords: torch.Tensor, 最后一维的4个数是坐标
    :param center: True表示将(xmin, ymin, xmax, ymax)转变为(center_x, center_y, w, h)格式
                   False表示将(xmin, ymin, xmax, ymax)转变为(xmin, ymin, w, h)格式
    :return: torch.Tensor, 与输入的形状一致
    """
    xmin = coords[..., 0:1]
    ymin = coords[..., 1:2]
    xmax = coords[..., 2:3]
    ymax = coords[..., 3:4]
    w = xmax - xmin
    h = ymax - ymin
    if center:
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        return torch.cat((center_x, center_y, w, h), dim=-1)
    else:
        return torch.cat((xmin, ymin, w, h), dim=-1)


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    box_a and box_b are both expected to be int (xmin, ymin, xmax, ymax) format.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def iou_2(anchors, boxes):
    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors)
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    anchor_max = anchors / 2
    anchor_min = - anchor_max
    box_max = boxes / 2
    box_min = - box_max
    intersect_min = torch.maximum(anchor_min, box_min)
    intersect_max = torch.minimum(anchor_max, box_max)
    intersect_wh = intersect_max - intersect_min
    intersect_wh = torch.clamp(intersect_wh, min=0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_area = boxes[..., 0] * boxes[..., 1]
    union_area = anchor_area + box_area - intersect_area
    iou = intersect_area / (union_area + 1e-12)  # shape : [N, 9]
    return iou


class Iou4:
    def __init__(self, box_1, box_2):
        """

        :param box_1: Tensor, shape: (..., 4(cx, cy, w, h))
        :param box_2: Tensor, shape: (..., 4(cx, cy, w, h))
        """
        self.box_1_min, self.box_1_max = Iou4._get_box_min_and_max(box_1)
        self.box_2_min, self.box_2_max = Iou4._get_box_min_and_max(box_2)
        self.box_1_area = box_1[..., 2] * box_1[..., 3]
        self.box_2_area = box_2[..., 2] * box_2[..., 3]

    @staticmethod
    def _get_box_min_and_max(box):
        box_xy = box[..., 0:2]
        box_wh = box[..., 2:4]
        box_min = box_xy - box_wh / 2
        box_max = box_xy + box_wh / 2
        return box_min, box_max

    def calculate_iou(self):
        intersect_min = torch.maximum(self.box_1_min, self.box_2_min)
        intersect_max = torch.minimum(self.box_1_max, self.box_2_max)
        intersect_wh = intersect_max - intersect_min
        intersect_wh = torch.clamp(intersect_wh, min=0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = self.box_1_area + self.box_2_area - intersect_area
        iou = intersect_area / (union_area + 1e-12)
        return iou


def truncate_array(a, n, use_padding=True, fill_value=-1):
    """
    对多维数组a在dim=0上截断
    :param a:
    :param n:
    :param use_padding: 是否将a的0维填充到n
    :param fill_value:  填充值，默认为-1
    :return:
    """
    if len(a) > n:
        return a[:n]
    else:
        if use_padding:
            shape = a.shape
            shape = (n,) + shape[1:]
            a = np.concatenate((a, np.full(shape, fill_value, dtype=a.dtype)), axis=0)
            return a
        else:
            return a


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchor center in gt
    判断哪些anchor_point在哪些gt_boxes的内部
    Args:
        xy_centers (Tensor): shape(8400, 4)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
    Return:
        (Tensor): shape(b, n_boxes, 8400)
    """
    n_anchors = xy_centers.shape[0]   # 8400
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
    # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
    out = bbox_deltas.amin(3).gt_(eps)  # (bs, n_boxes, 8400)  dtype=torch.float32
    return out


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    # (b, n_max_boxes, h*w) -> (b, h*w)
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])  # (b, n_max_boxes, h*w)
        max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)  # (b, h*w, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)  # (b, n_max_boxes, h*w)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)  # (b, n_max_boxes, h*w)
        fg_mask = mask_pos.sum(-2)
    # Find each grid serve which gt(index)
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric,
    which combines both classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.
        Reference https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            # 默认类别序号是80
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt)

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """

        :param pd_scores: Tensor, (bs, num_total_anchors, num_classes)
        :param pd_bboxes: Tensor, (bs, num_total_anchors, 4)
        :param gt_labels: Tensor, (bs, n_max_boxes, 1)
        :param gt_bboxes: Tensor, (bs, n_max_boxes, 4)
        :param anc_points: Tensor, (num_total_anchors, 2)
        :param mask_gt: Tensor, (bs, n_max_boxes, 1)
        :return:
        """
        # (bs, n_max_boxes, num_total_anchors)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, 8400)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, 8400)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        # Merge all mask to a final mask, (b, max_num_obj, 8400)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        综合考虑分类score和iou
        :param pd_scores: Tensor, (bs, num_total_anchors, num_classes)
        :param pd_bboxes: Tensor, (bs, num_total_anchors, 4)
        :param gt_labels: Tensor, (bs, n_max_boxes, 1)
        :param gt_bboxes: Tensor, (bs, n_max_boxes, 4)
        :param mask_gt: Tensor, (bs, n_max_boxes, num_total_anchors), dtype=torch.float32
        :return:
        """
        na = pd_bboxes.shape[-2]  # 8400
        mask_gt = mask_gt.bool()  # b, max_num_obj, 8400
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.long().squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, 8400

        pd_boxes = pd_bboxes.unsqueeze(1).repeat(1, self.n_max_boxes, 1, 1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).repeat(1, 1, na, 1)[mask_gt]
        # (b, max_num_obj, 8400)
        overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp(0)
        # (b, max_num_obj, 8400)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, 8400), where b is the batch size,
                              max_num_obj is the maximum number of objects, and 8400 represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """

        num_anchors = metrics.shape[-1]  # h*w
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        # (b, max_num_obj, topk)
        topk_idxs[~topk_mask] = 0
        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        is_in_topk = torch.zeros(metrics.shape, dtype=torch.long, device=metrics.device)
        for it in range(self.topk):
            is_in_topk += F.one_hot(topk_idxs[:, :, it], num_anchors)
        # is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        # filter invalid bboxes
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """

        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # Assigned target scores
        target_labels.clamp(0)
        target_scores = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores