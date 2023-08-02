import itertools
from math import sqrt

import numpy as np
import torch


def generate_ssd_anchor(input_image_shape, anchor_sizes, feature_shapes, aspect_ratios):
    """
    生成SSD算法需要的锚框
    :param input_image_shape: 输入图片高、宽
    :param anchor_sizes:
    :param feature_shapes: 输入特征图的高、宽
    :param aspect_ratios:
    :return:
    """
    image_h, image_w = input_image_shape
    anchors = []
    for i, s in enumerate(feature_shapes):
        sk1 = anchor_sizes[i] / image_w
        sk2 = anchor_sizes[i + 1] / image_h
        sk3 = sqrt(sk1 * sk2)
        all_sizes = [(sk1, sk1), (sk3, sk3)]

        for ar in aspect_ratios[i]:
            if ar != 1:
                all_sizes.append((sk1 * sqrt(ar), sk1 / sqrt(ar)))

        for w, h in all_sizes:
            for m, n in itertools.product(range(s), repeat=2):
                cx, cy = (n + 0.5) / feature_shapes[i], (m + 0.5) / feature_shapes[i]
                anchors.append((cx, cy, w, h))

    anchors = np.array(anchors, dtype=np.float32)
    anchors = np.clip(anchors, a_min=0, a_max=1)
    anchors_ltrb = anchors.copy()  # (xmin, ymin, xmax, ymax)格式
    anchors_ltrb[:, 0] = anchors[:, 0] - 0.5 * anchors[:, 2]
    anchors_ltrb[:, 1] = anchors[:, 1] - 0.5 * anchors[:, 3]
    anchors_ltrb[:, 2] = anchors[:, 0] + 0.5 * anchors[:, 2]
    anchors_ltrb[:, 3] = anchors[:, 1] + 0.5 * anchors[:, 3]

    return anchors_ltrb


def generate_ssd_anchor_v2(input_image_shape, anchor_sizes, feature_shapes, aspect_ratios):
    image_h, image_w = input_image_shape
    anchors = []
    for i in range(len(feature_shapes)):
        # 先验框的短边和长边
        min_size = anchor_sizes[i]
        max_size = anchor_sizes[i + 1]
        # 特征图的高和宽，它们相等
        feature_h = feature_shapes[i]
        # 对于每个像素位置，根据aspect_ratio生成不同宽、高比的先验框
        box_widths = []
        box_heights = []
        for ar in aspect_ratios[i]:
            if ar == 1:
                box_widths.append(min_size)
                box_heights.append(min_size)
                box_widths.append(np.sqrt(min_size * max_size))
                box_heights.append(np.sqrt(min_size * max_size))
            else:
                box_widths.append(min_size * np.sqrt(ar))
                box_heights.append(min_size / np.sqrt(ar))

        half_box_widths = np.array(box_widths) / 2.0  # shape: (len(aspect_ratios[i])+1,)
        half_box_heights = np.array(box_heights) / 2.0

        # 特征层上一个像素点映射到原图上对应的像素长度
        pixel_length = [image_h / feature_h, image_w / feature_h]
        # 生成网格中心
        c_x = np.linspace(0.5 * pixel_length[1], image_w - 0.5 * pixel_length[1], feature_h)
        c_y = np.linspace(0.5 * pixel_length[0], image_h - 0.5 * pixel_length[0], feature_h)
        center_x, center_y = np.meshgrid(c_x, c_y)
        center_x = np.reshape(center_x, (-1, 1))  # (feature_h**2, 1)
        center_y = np.reshape(center_y, (-1, 1))  # (feature_h**2, 1)

        anchor = np.concatenate((center_x, center_y), axis=1)  # (feature_h**2, 2)
        # 对于每一种宽高比例，都需要一个对应的先验框
        # shape: (feature_h**2, 4*(len(aspect_ratios[i])+1))
        anchor = np.tile(anchor, (1, (len(aspect_ratios[i]) + 1) * 2))

        # 转换为xmin, ymin, xmax, ymax格式
        anchor[:, ::4] -= half_box_widths  # shape: (feature_h**2, len(aspect_ratios[i])+1)
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


def generate_yolo3_anchor(cfg, device, idx=None):
    c, h, w = cfg.arch.input_size
    anchors = cfg.arch.anchor
    anchors = torch.tensor(anchors, dtype=torch.float32)
    anchors = torch.reshape(anchors, shape=(-1, 2))
    # 归一化
    anchors[:, 0] /= w
    anchors[:, 1] /= h

    if device is not None:
        anchors = anchors.to(device)

    if idx is None:
        return anchors
    else:
        return anchors[3 * idx: 3 * (idx + 1), :]


def get_yolo7_anchors(cfg):
    anchors_list = cfg.arch.anchors
    anchors = np.array(anchors_list, dtype=np.float32).reshape(-1, 2)
    return anchors


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    生成YOLOv8的anchor
    :param feats:
    :param strides:
    :param grid_cell_offset:
    :return:
    """

    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


