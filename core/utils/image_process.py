import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from core.utils.bboxes import xywh_to_xyxy


def read_image(image_path, mode='rgb'):
    """
    使用opencv读取图像
    :param image_path: 图像文件路径
    :param mode: 格式，'rgb', 'bgr', 'gray'
    :return: numpy.ndarray, dtype=np.uint8, shape=(h, w, c)
    """

    assert mode in ['rgb', 'bgr', 'gray'], "mode must be one of 'rgb', 'bgr', 'gray'"
    image_array = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if mode == "rgb":
        return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    elif mode == 'gray':
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, axis=-1)
        return gray
    else:
        return image_array


def read_image_and_convert_to_tensor(image_path, size, mode='rgb', letterbox=True):
    """
    使用opencv读取图像，并进行resize，之后转换为torch.Tensor
    :param image_path: 图像文件路径
    :param size: [h, w]
    :param mode: 格式，'rgb', 'bgr', 'gray'
    :param letterbox: 是否使用保持宽高比的resize方式
    :return: [torch.Tensor, shape=(1, c, h, w)], h, w
    """
    image_array = read_image(image_path, mode)
    h, w, _ = image_array.shape
    if letterbox:
        image_array, _, _ = letter_box(image_array, size)
    else:
        image_array = cv2.resize(src=image_array, dsize=size[::-1], interpolation=cv2.INTER_CUBIC)
    image = TF.to_tensor(image_array).unsqueeze(0)
    return image, h, w


def letter_box(image, size):
    """
    resize图片的同时保持宽高比不变
    :param image: numpy.ndarray
    :param size: list or tuple (h, w)
    :return:
    """
    h, w, _ = image.shape
    H, W = size
    scale = min(H / h, W / w)
    new_h, new_w = int(h * scale), int(w * scale)
    image = cv2.resize(src=image, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
    top = (H - new_h) // 2  # 上边需要填充的像素数量
    bottom = H - new_h - top  # 下边需要填充的像素数量
    left = (W - new_w) // 2  # 左边需要填充的像素数量
    right = W - new_w - left  # 右边需要填充的像素数量
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT,
                                   value=(128, 128, 128))
    return new_image, scale, [top, bottom, left, right]


def reverse_letter_box_numpy(image_shape, input_shape, boxes, xywh=True):
    """
    letter_box的逆变换
    :param image_shape: 输入网络的图片的原始形状 [h, w]
    :param input_shape: List or Tuple 网络输入图片的固定大小 [H, W]
    :param boxes: numpy.ndarray, shape: (..., 4(cx, cy, w, h))
    :param xywh: Bool, True：boxes是(cx, cy, w, h)格式, False: boxes是(xmin, ymin, xmax, ymax)格式
    :return: numpy.ndarray, shape: (..., 4(xmin, ymin, xmax, ymax))
    """
    # 转换为(xmin, ymin, xmax, ymax)格式
    if xywh:
        new_boxes = np.concatenate((boxes[..., 0:2] - boxes[..., 2:4] / 2, boxes[..., 0:2] + boxes[..., 2:4] / 2), axis=-1)
    else:
        new_boxes = boxes.copy()
    new_boxes[..., ::2] *= input_shape[1]
    new_boxes[..., 1::2] *= input_shape[0]

    scale = max(image_shape[0] / input_shape[0], image_shape[1] / input_shape[1])
    # 获取padding值
    top = (input_shape[0] - image_shape[0] / scale) // 2
    left = (input_shape[1] - image_shape[1] / scale) // 2
    # 减去padding值，就是相对于原始图片的原点位置
    new_boxes[..., 0] -= left
    new_boxes[..., 2] -= left
    new_boxes[..., 1] -= top
    new_boxes[..., 3] -= top
    # 缩放到原图尺寸
    new_boxes *= scale
    return new_boxes


def reverse_letter_box(h, w, input_size, boxes, xywh=True):
    """
    letter_box的逆变换
    :param h: 输入网络的图片的原始高度
    :param w: 输入网络的图片的原始宽度
    :param input_size: List or Tuple 网络输入图片的固定大小
    :param boxes: Tensor, shape: (..., 4(cx, cy, w, h))
    :param xywh: Bool, True：boxes是(cx, cy, w, h)格式, False: boxes是(xmin, ymin, xmax, ymax)格式
    :return: Tensor, shape: (..., 4(xmin, ymin, xmax, ymax))
    """
    # 转换为(xmin, ymin, xmax, ymax)格式
    if xywh:
        new_boxes = torch.cat((boxes[..., 0:2] - boxes[..., 2:4] / 2, boxes[..., 0:2] + boxes[..., 2:4] / 2), dim=-1)
    else:
        new_boxes = boxes.clone()
    new_boxes[..., ::2] *= input_size[1]
    new_boxes[..., 1::2] *= input_size[0]

    scale = max(h / input_size[0], w / input_size[1])
    # 获取padding值
    top = (input_size[0] - h / scale) // 2
    left = (input_size[1] - w / scale) // 2
    # 减去padding值，就是相对于原始图片的原点位置
    new_boxes[..., 0] -= left
    new_boxes[..., 2] -= left
    new_boxes[..., 1] -= top
    new_boxes[..., 3] -= top
    # 缩放到原图尺寸
    new_boxes *= scale
    return new_boxes


def cv2_paste(img1, img2, x, y):
    """
    实现Pillow库中的paste函数
    :param img1: 背景图片
    :param img2: 第二张图片
    :param x: 左上角的x坐标
    :param y: 左上角的y坐标
    :return:
    """
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    # 计算img1和img2在同一直角坐标系下的相交区域
    xmin = max(x, 0)
    ymin = max(y, 0)
    xmax = min(w1, x + w2)
    ymax = min(h1, y + h2)

    # 计算这个相交区域在img2上的相对坐标
    xmin_ = xmin - x
    xmax_ = xmax - x
    ymin_ = ymin - y
    ymax_ = ymax - y

    # 把这个相交区域覆盖到img1上
    img1[ymin:ymax, xmin:xmax, :] = img2[ymin_:ymax_, xmin_:xmax_, :]

    return img1


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    """

    :param box_xy:
    :param box_wh:
    :param input_shape: 网络的固定输入图片尺寸
    :param image_shape: 真实的输入图片大小
    :param letterbox_image:
    :return:
    """
    box_xywh = np.concatenate([box_xy, box_wh], axis=-1)
    if letterbox_image:
        return reverse_letter_box_numpy(image_shape, input_shape,
                                        box_xywh,
                                        xywh=True)
    else:
        # 转换为(xmin, ymin, xmax, ymax)格式
        box_x1y1x2y2 = xywh_to_xyxy(box_xywh)
        box_x1y1x2y2[:, ::2] *= image_shape[1]
        box_x1y1x2y2[:, 1::2] *= image_shape[0]
        return box_x1y1x2y2
