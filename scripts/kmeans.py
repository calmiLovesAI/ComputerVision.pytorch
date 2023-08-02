import torch

import numpy as np

from core.data import transforms as T

from tqdm import tqdm
from core.data.voc import Voc
from torch.utils.data import DataLoader

# 配置文件
CFG = {
    "input_size": [608, 608],
    "dataset": "voc",    # "voc" or "coco"
    "batch_size": 8,
    "max_num_boxes": 20
}


def build_dataloader(cfg):
    if cfg["dataset"] == "voc":
        dataset = Voc(True, T.Compose(transforms=[
            T.Resize(size=cfg["input_size"]),
            T.TargetPadding(max_num_boxes=cfg["max_num_boxes"]),
            T.ToTensor()
        ]))
    else:
        raise ValueError("参数dataset错误")
    return DataLoader(dataset=dataset, batch_size=cfg["batch_size"], shuffle=True)


class KMeans:
    def __init__(self, cfg, dataloader):
        self.cfg = cfg
        self.dataloader = dataloader
        self.input_size = cfg["input_size"]

    def __call__(self, k):
        boxes = self._load_data().numpy()
        row = boxes.shape[0]
        distance = np.empty((row, k))
        last_cluster = np.zeros((row,))
        # np.random.seed()
        cluster = boxes[np.random.choice(row, k, replace=False)]
        while True:
            for i in range(row):
                distance[i] = 1 - KMeans.get_iou(boxes[i], cluster)

            near = np.argmin(distance, axis=1)

            if (last_cluster == near).all():
                break

            for j in range(k):
                cluster[j] = np.median(boxes[near == j], axis=0)

            last_cluster = near
        anchors = cluster
        anchors = anchors[np.argsort(anchors[:, 0])]
        acc = KMeans.average_iou(boxes, anchors)
        print("Accuracy: {:.2f}%".format(acc * 100))
        anchors = (anchors * self.input_size).astype(np.int32)
        return anchors

    @staticmethod
    def get_iou(box, cluster):
        x = np.minimum(box[0], cluster[:, 0])
        y = np.minimum(box[1], cluster[:, 1])

        inter = x * y
        area_box = box[0] * box[1]
        area_cluster = cluster[:, 0] * cluster[:, 1]
        iou = inter / (area_box + area_cluster - inter)
        return iou

    @staticmethod
    def average_iou(boxes, cluster):
        return np.mean([np.max(KMeans.get_iou(boxes[i], cluster)) for i in range(boxes.shape[0])])

    def _load_data(self):
        boxes = list()
        for _, tar in tqdm(self.dataloader, desc="读取数据集"):
            tar = torch.reshape(tar, shape=(-1, 5))
            tar = tar[tar[:, -1] != -1]
            box_xyxy = tar[:, :4]
            box_wh = torch.stack((box_xyxy[:, 2] - box_xyxy[:, 0], box_xyxy[:, 3] - box_xyxy[:, 1]), dim=-1)
            boxes.append(box_wh)
        return torch.cat(boxes, dim=0)


if __name__ == '__main__':
    k_means = KMeans(CFG, build_dataloader(CFG))
    print(k_means(k=9))
