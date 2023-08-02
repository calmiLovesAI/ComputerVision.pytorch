import torch

from configs.yolov3_cfg import Config
from core.utils.anchor import generate_yolo3_anchor
from core.utils.image_process import read_image, letter_box, reverse_letter_box
from core.utils.nms import yolo3_nms
import torchvision.transforms.functional as TF

from core.utils.visualize import show_detection_results


def predict_bounding_bbox(num_classes, feature_map, anchors, device, is_training=False):
    N, C, H, W = feature_map.size()
    feature_map = torch.permute(feature_map, dims=(0, 2, 3, 1))
    anchors = torch.reshape(anchors, shape=(1, 1, 1, -1, 2))
    grid_y = torch.reshape(torch.arange(0, H, dtype=torch.float32, device=device), (-1, 1, 1, 1))
    grid_y = torch.tile(grid_y, dims=(1, W, 1, 1))
    grid_x = torch.reshape(torch.arange(0, W, dtype=torch.float32, device=device), (1, -1, 1, 1))
    grid_x = torch.tile(grid_x, dims=(H, 1, 1, 1))
    grid = torch.cat((grid_x, grid_y), dim=-1)
    feature_map = torch.reshape(feature_map, shape=(-1, H, W, 3, num_classes + 5))
    box_xy = (torch.sigmoid(feature_map[..., 0:2]) + grid) / H
    box_wh = torch.exp(feature_map[..., 2:4]) * anchors
    confidence = torch.sigmoid(feature_map[..., 4:5])
    class_prob = torch.sigmoid(feature_map[..., 5:])
    if is_training:
        return box_xy, box_wh, grid, feature_map
    else:
        return box_xy, box_wh, confidence, class_prob


class Decoder:
    def __init__(self, cfg, conf_threshold, device):
        self.cfg = cfg
        self.device = device
        self.num_classes = cfg.arch.num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = cfg.decode.iou_threshold

    def _yolo_post_process(self, feature, scale_type):
        box_xy, box_wh, confidence, class_prob = predict_bounding_bbox(self.num_classes, feature,
                                                                       generate_yolo3_anchor(self.cfg, self.device, scale_type),
                                                                       self.device, is_training=False)
        # boxes = reverse_letter_box(self.input_image_h, self.input_image_w, self.cfg["Train"]["input_size"],
        #                            torch.cat((box_xy, box_wh), dim=-1))
        # (xmin, ymin, xmax, ymax)
        boxes = torch.cat((box_xy - box_wh / 2, box_xy + box_wh / 2), dim=-1)
        boxes = torch.reshape(boxes, shape=(-1, 4))
        boxes_scores = confidence * class_prob
        boxes_scores = torch.reshape(boxes_scores, shape=(-1, self.num_classes))
        return boxes, boxes_scores

    def __call__(self, outputs):
        boxes_list = list()
        boxes_scores_list = list()
        for i in range(3):
            boxes, boxes_scores = self._yolo_post_process(feature=outputs[i],
                                                          scale_type=i)
            boxes_list.append(boxes)
            boxes_scores_list.append(boxes_scores)
        boxes = torch.cat(boxes_list, dim=0)
        scores = torch.cat(boxes_scores_list, dim=0)
        boxes, scores, classes = yolo3_nms(self.num_classes, self.conf_threshold, self.iou_threshold, boxes, scores,
                                           self.device)
        scores = torch.squeeze(scores, dim=-1)
        return boxes, scores, classes


def detect_one_image(cfg: Config, model, image_path, print_on, save_result, device):
    model.eval()
    # 处理单张图片
    image = read_image(image_path)
    h, w, c = image.shape
    image, _, _ = letter_box(image, cfg.arch.input_size[1:])
    image = TF.to_tensor(image).unsqueeze(0)
    image = image.to(device)

    decoder = Decoder(cfg,
                      conf_threshold=cfg.decode.conf_threshold,
                      device=device)

    with torch.no_grad():
        preds = model(image)
        boxes, scores, classes = decoder(preds)
        # 将boxes坐标变换到原始图片上
        boxes = reverse_letter_box(h=h, w=w, input_size=cfg.arch.input_size[1:], boxes=boxes, xywh=False)

    show_detection_results(image_path=image_path,
                           dataset_name=cfg.dataset.dataset_name,
                           boxes=boxes,
                           scores=scores,
                           class_indices=classes,
                           print_on=print_on,
                           save_result=save_result,
                           save_dir=cfg.decode.test_results)
