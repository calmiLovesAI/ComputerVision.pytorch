import os

import numpy as np
import torch
import xml.etree.ElementTree as ET
from tqdm import tqdm

from core.data import get_voc_root_and_classes
from core.metrics.mAP import get_map, get_coco_map
from core.utils.image_process import read_image, letter_box, reverse_letter_box
import torchvision.transforms.functional as TF


def evaluate_pipeline(model,
                      decoder,
                      input_image_size,
                      map_out_root,
                      dataset='voc',
                      subset='val',
                      list_input=False,
                      device=None,
                      skip=False):
    """
    验证模型性能的完整管道
    :param model: 网络模型
    :param decoder: 此模型对应的解码器
    :param input_image_size:  [h, w], 模型输入图片大小
    :param map_out_root:  检测结果保存的根目录
    :param dataset:
    :param subset: 子集，'val' or 'test'
    :param list_input: decoder解码后的输出是list
    :param device: 设备
    :param skip: 跳过生成图片检测文件和gt文件的步骤
    :return:
    """
    gt_path = os.path.join(map_out_root, 'ground-truth')
    detections_path = os.path.join(map_out_root, 'detection-results')
    images_optional_path = os.path.join(map_out_root, 'images-optional')

    for p in [map_out_root, gt_path, detections_path, images_optional_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    voc_root, voc_class_names = get_voc_root_and_classes()
    if subset == 'val':
        image_ids = open(os.path.join(voc_root, "ImageSets", "Main", "val.txt"), mode='r').read().strip().split(
            '\n')
    elif subset == 'test':
        image_ids = open(os.path.join(voc_root, "ImageSets", "Main", "test.txt"), mode='r').read().strip().split(
            '\n')
    else:
        raise ValueError(f"sub_set must be one of 'test' and 'val', but got {subset}")

    if not skip:
        with tqdm(image_ids, desc=f"Evaluate on voc-{subset}") as pbar:
            for image_id in pbar:
                # 图片预处理
                image_path = os.path.join(voc_root, "JPEGImages", f"{image_id}.jpg")
                image = read_image(image_path)
                h, w, c = image.shape
                image, _, _ = letter_box(image, input_image_size)
                image = TF.to_tensor(image).unsqueeze(0)
                image = image.to(device)

                # 得到检测结果
                if list_input:
                    decoder.set_h_w(h, w)
                    with torch.no_grad():
                        preds = model(image)
                        results = decoder(preds)

                    if len(results[0]) == 0:
                        # 填充0
                        results.append(np.array([0, 0, 0, 0, 0, 0], dtype=np.float32))

                    boxes = results[0][:, :4]
                    scores = results[0][:, 5]
                    class_indices = results[0][:, 4].astype(np.int32)
                else:
                    with torch.no_grad():
                        preds = model(image)
                        boxes, scores, classes = decoder(preds)
                        # 将boxes坐标变换到原始图片上
                        boxes = reverse_letter_box(h=h, w=w, input_size=input_image_size, boxes=boxes, xywh=False)

                    boxes = boxes.cpu().numpy()
                    scores = scores.cpu().numpy()
                    class_indices = classes.cpu().numpy().tolist()

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
