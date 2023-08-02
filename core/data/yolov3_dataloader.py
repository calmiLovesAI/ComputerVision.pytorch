from configs.yolov3_cfg import Config
from core.data.dataloader import PublicDataLoader
from core.data.transforms import TargetPadding


class Yolo3Loader(PublicDataLoader):
    def __init__(self, cfg: Config, dataset_name: str, batch_size, input_size):
        super().__init__(dataset_name, batch_size, input_size)
        self.train_transforms.insert(1, TargetPadding(max_num_boxes=cfg.train.max_num_boxes))