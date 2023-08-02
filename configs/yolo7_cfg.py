from registry import config_registry
from .dataset_cfg import VOC_CFG, COCO_CFG


@config_registry("yolo7")
class Yolo7Config:
    def __init__(self):
        self.arch = self._Arch()
        self.dataset = self._Dataset()
        self.train = self._Train()
        self.loss = self._Loss()
        self.optimizer = self._Optimizer()
        self.log = self._Log()
        self.decode = self._Decode()

    class _Arch:
        def __init__(self):
            # 输入图片大小：(C, H, W)
            self.input_size = (3, 640, 640)
            self.anchors = [
                12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
            ]
            # anchor的编号
            self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            # 用到的yolov7的版本，l: yolov7, x: yolov7_x
            self.phi = 'l'

    class _Dataset:
        # 数据集
        def __init__(self):
            # 数据集名称，"voc"或者"coco"
            self.dataset_name = VOC_CFG["name"]
            # 目标类别数，与数据集有关，对于voc是20，对于coco是80
            self.num_classes = VOC_CFG["num_classes"]

    class _Train:
        # 训练参数
        def __init__(self):
            # 恢复训练时加载的checkpoint文件，None表示从epoch=0开始训练
            self.resume_training = ""
            # 恢复训练时的上一次epoch是多少，-1表示从epoch=0开始训练
            self.last_epoch = -1

            self.epoch = 100
            self.batch_size = 4
            # 初始学习率
            self.initial_lr = 1e-3
            # warm up轮数，设为0表示不使用warm up
            self.warmup_iters = 0
            self.milestones = [30, 60]
            self.gamma = 0.1

            # 是否使用预训练权重
            self.pretrained = True
            # 预训练模型的权重路径
            self.pretrained_weights = "saves/yolov7_weights.pth"
            # 模型保存间隔
            self.save_interval = 5
            # 每隔多少epoch在验证集上验证一次，0表示不验证
            self.eval_interval = 0
            # 保存模型的文件夹
            self.save_path = "saves"
            # 是否启动tensorboard
            self.tensorboard_on = True
            # 是否使用混合精度训练
            self.mixed_precision = True
            # 多少个子进程用于数据加载
            self.num_workers = 0
            self.max_num_boxes = 30

    class _Loss:
        # 损失函数
        def __init__(self):
            self.ignore_threshold = 0.5
            # 标签平滑，一般在0.01以下
            self.label_smoothing = 0

    class _Optimizer:
        # 优化器
        def __init__(self):
            self.name = "Adam"
            self.scheduler_name = "multi_step"

    class _Log:
        # 训练日志
        def __init__(self):
            # 日志文件保存文件夹
            self.root = "log"
            # 日志文件输出间隔
            self.print_interval = 50

    class _Decode:
        def __init__(self):
            self.test_results = "result"
            # 是否使用letterbox的方式对图片进行预处理
            self.letterbox_image = True
            self.conf_threshold = 0.5
            self.nms_threshold = 0.3
