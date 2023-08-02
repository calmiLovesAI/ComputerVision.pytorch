import os
import time

import cv2
import numpy as np

from core.data import find_class_name
from core.utils.image_process import read_image


def now():
    return time.strftime("%Y-%m-%d-%H-%M-%S")


def show_detection_results(image_path, dataset_name, boxes, scores, class_indices,
                           print_on=True, save_result=True, save_dir=None):
    """
    可视化目标检测结果
    :param image_path: 原始图片路径
    :param dataset_name: 数据集名称
    :param boxes: numpy.ndarray, shape: (N, 4)
    :param scores:  numpy.ndarray, shape: (N,)
    :param class_indices:  numpy.ndarray, shape: (N,)
    :param print_on: 是否在控制台打印检测框的信息
    :param save_result: 是否保存检测结果
    :param save_dir: 检测结果保存的文件夹
    :return:
    """
    # 移除坐标不在图片大小范围内的检测框
    ori_image = read_image(image_path, mode='bgr')
    # ori_h, ori_w, _ = ori_image.shape
    # mask = torch.logical_and((boxes[:, 0] > 0), (boxes[:, 1] > 0))
    # mask = torch.logical_and(mask, (boxes[:, 2] < ori_w))
    # mask = torch.logical_and(mask, (boxes[:, 3] < ori_h))
    # boxes = boxes[mask]
    # scores = scores[mask]
    # class_indices = class_indices[mask]
    n = boxes.shape[0]
    if n == 0:
        # 没有检测到目标
        if print_on:
            print("Detect 0 object")
        image_with_boxes = ori_image
    else:
        if print_on:
            print(f"Detect {n} objects: ")
        class_indices = class_indices.tolist()
        class_names = [find_class_name(dataset_name, c, keep_index=False) for c in class_indices]

        if print_on:
            print("boxes: ", boxes)
            print("scores: ", scores)
            print("classes: ", class_names)

        painter = Draw()
        image_with_boxes = painter.draw_boxes_on_image(ori_image, boxes, scores, class_ids=class_indices,
                                                       class_names=class_names)
    if save_result:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_filename = os.path.join(save_dir, os.path.basename(image_path).split(".")[0] + f"@{now()}.jpg")
        # 保存检测结果
        cv2.imwrite(save_filename, image_with_boxes)
    else:
        return image_with_boxes


class Draw:
    def __init__(self):
        self.colors = np.array(
            [
                0.000, 0.447, 0.741,
                0.850, 0.325, 0.098,
                0.929, 0.694, 0.125,
                0.494, 0.184, 0.556,
                0.466, 0.674, 0.188,
                0.301, 0.745, 0.933,
                0.635, 0.078, 0.184,
                0.300, 0.300, 0.300,
                0.600, 0.600, 0.600,
                1.000, 0.000, 0.000,
                1.000, 0.500, 0.000,
                0.749, 0.749, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 1.000,
                0.667, 0.000, 1.000,
                0.333, 0.333, 0.000,
                0.333, 0.667, 0.000,
                0.333, 1.000, 0.000,
                0.667, 0.333, 0.000,
                0.667, 0.667, 0.000,
                0.667, 1.000, 0.000,
                1.000, 0.333, 0.000,
                1.000, 0.667, 0.000,
                1.000, 1.000, 0.000,
                0.000, 0.333, 0.500,
                0.000, 0.667, 0.500,
                0.000, 1.000, 0.500,
                0.333, 0.000, 0.500,
                0.333, 0.333, 0.500,
                0.333, 0.667, 0.500,
                0.333, 1.000, 0.500,
                0.667, 0.000, 0.500,
                0.667, 0.333, 0.500,
                0.667, 0.667, 0.500,
                0.667, 1.000, 0.500,
                1.000, 0.000, 0.500,
                1.000, 0.333, 0.500,
                1.000, 0.667, 0.500,
                1.000, 1.000, 0.500,
                0.000, 0.333, 1.000,
                0.000, 0.667, 1.000,
                0.000, 1.000, 1.000,
                0.333, 0.000, 1.000,
                0.333, 0.333, 1.000,
                0.333, 0.667, 1.000,
                0.333, 1.000, 1.000,
                0.667, 0.000, 1.000,
                0.667, 0.333, 1.000,
                0.667, 0.667, 1.000,
                0.667, 1.000, 1.000,
                1.000, 0.000, 1.000,
                1.000, 0.333, 1.000,
                1.000, 0.667, 1.000,
                0.333, 0.000, 0.000,
                0.500, 0.000, 0.000,
                0.667, 0.000, 0.000,
                0.833, 0.000, 0.000,
                1.000, 0.000, 0.000,
                0.000, 0.167, 0.000,
                0.000, 0.333, 0.000,
                0.000, 0.500, 0.000,
                0.000, 0.667, 0.000,
                0.000, 0.833, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 0.167,
                0.000, 0.000, 0.333,
                0.000, 0.000, 0.500,
                0.000, 0.000, 0.667,
                0.000, 0.000, 0.833,
                0.000, 0.000, 1.000,
                0.000, 0.000, 0.000,
                0.143, 0.143, 0.143,
                0.286, 0.286, 0.286,
                0.429, 0.429, 0.429,
                0.571, 0.571, 0.571,
                0.714, 0.714, 0.714,
                0.857, 0.857, 0.857,
                0.000, 0.447, 0.741,
                0.314, 0.717, 0.741,
                0.50, 0.5, 0
            ]
        ).astype(np.float32).reshape(-1, 3)
        # # r, g, b
        # self.colors = {
        #     "粉红": (255, 192, 203),
        #     "红色": (255, 0, 0),
        #     "紫罗兰": (238, 130, 238),
        #     "洋红": (255, 0, 255),
        #     "深天蓝": (0, 191, 255),
        #     "青色": (0, 255, 255),
        #     "春天的绿色": (60, 179, 113),
        #     "浅海洋绿": (32, 178, 170),
        #     "米色": (245, 245, 220),
        #     "小麦色": (245, 222, 179),
        #     "棕色": (165, 42, 42),
        #     "深灰色": (169, 169, 169),
        #     "黄色": (255, 255, 255),
        #     "紫红色": (255, 0, 255)
        # }

    def draw_boxes_on_image(self, image, boxes, scores, class_ids, class_names):
        # image = cv2.imread(image_path)
        # h, w, _ = image.shape
        # d, r = self._get_adaptive_zoom_ratio(h, w)
        boxes = boxes.astype(int)

        num_boxes = boxes.shape[0]
        for i in range(num_boxes):
            cls_id = class_ids[i]
            score = scores[i]
            x0 = boxes[i, 0]
            y0 = boxes[i, 1]
            x1 = boxes[i, 2]
            y1 = boxes[i, 3]
            color = (self.colors[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[i], score * 100)
            txt_color = (0, 0, 0) if np.mean(self.colors[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (self.colors[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                image,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(image, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

            # class_and_score = classes[i][0] + ": {:.2f}".format(scores[i])
            # # 获取类别对应的颜色
            # bbox_color = self._get_rgb_color(classes[i][1])
            # bbox_color_bgr = bbox_color[::-1]
            # cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]),
            #               color=bbox_color_bgr,
            #               thickness=2)
            # cv2.putText(img=image, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - int(d)),
            #             fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=r, color=(0, 255, 255), thickness=1)
        return image



def show_supported_models_on_command_line(model_registry):
    """
    在命令行中显示支持的模型
    :param registry: 模型注册器
    :return:
    """
    print("===========================")
    print("Supported models: ")
    for i, item in enumerate(model_registry.keys()):
        # 移除开头的 "model_"
        model_name = item[6:]
        print(f"{i}: {model_name}")
    print("===========================")