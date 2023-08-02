from .dataset_cfg import COCO_CFG, VOC_CFG
from registry import config_registry


def get_ar(input_size: int):
    """
    :param input_size:
    :return:
    """
    if input_size == 300:
        return [
            [1, 2, 1.0 / 2],
            [1, 2, 1.0 / 2, 3, 1.0 / 3],
            [1, 2, 1.0 / 2, 3, 1.0 / 3],
            [1, 2, 1.0 / 2, 3, 1.0 / 3],
            [1, 2, 1.0 / 2],
            [1, 2, 1.0 / 2],
        ]
    else:
        return [
            [1, 2, 1.0 / 2],
            [1, 2, 1.0 / 2, 3, 1.0 / 3],
            [1, 2, 1.0 / 2, 3, 1.0 / 3],
            [1, 2, 1.0 / 2, 3, 1.0 / 3],
            [1, 2, 1.0 / 2, 3, 1.0 / 3],
            [1, 2, 1.0 / 2],
            [1, 2, 1.0 / 2],
        ]


def get_feature_shapes(input_size: int):
    if input_size == 300:
        return [38, 19, 10, 5, 3, 1]
    else:
        return [64, 32, 16, 8, 4, 2, 1]


def get_feature_channels(input_size: int):
    if input_size == 300:
        return [512, 1024, 512, 256, 256, 256]
    else:
        return [512, 1024, 512, 256, 256, 256, 256]


def get_anchor_sizes(input_size: int):
    if input_size == 300:
        return [30, 60, 111, 162, 213, 264, 315]
    else:
        return [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72]


def check_input_size(input_size):
    """
    检查输入图片大小是否符合要求，输入图片大小必须是300或者512，
    这是因为SSD的网络结构与输入图片大小有关
    """
    assert isinstance(input_size, tuple)
    assert len(input_size) == 3
    assert input_size[2] == input_size[1]
    assert input_size[1] == 300 or input_size[1] == 512


@config_registry("ssd")
class SsdConfig:
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
            self.backbone = "vgg"
            # 输入图片大小：(3, 300, 300) 或 (3, 512, 512)
            self.input_size = (3, 300, 300)
            check_input_size(self.input_size)

            # 先验框的宽高比
            self.aspect_ratios = get_ar(self.input_size[1])
            # ssd结构中6个特征层的输出通道数
            self.feature_channels = get_feature_channels(self.input_size[1])
            self.feature_shapes = get_feature_shapes(self.input_size[1])
            # 先验框的宽和高
            self.anchor_sizes = get_anchor_sizes(self.input_size[1])

    class _Dataset:
        # 数据集
        def __init__(self):
            # 目标类别数，与数据集有关，对于voc是20，对于coco是80
            self.num_classes = VOC_CFG["num_classes"]
            # 数据集名称，"voc"或者"coco"
            self.dataset_name = VOC_CFG["name"]

    class _Train:
        # 训练参数
        def __init__(self):
            # 恢复训练时加载的checkpoint文件，""表示从epoch=0开始训练
            self.resume_training = ""
            # 恢复训练时的上一次epoch是多少，-1表示从epoch=0开始训练
            self.last_epoch = -1

            self.epoch = 100
            self.batch_size = 16
            # 初始学习率
            self.initial_lr = 1e-3
            # warm up轮数
            self.warmup_iters = 1000
            self.milestones = []
            self.gamma = 0.1

            # 是否使用预训练权重
            self.pretrained = False
            # 预训练模型的权重路径
            self.pretrained_weights = ""
            # 模型保存间隔
            self.save_interval = 1
            # 每隔多少epoch在验证集上验证一次
            self.eval_interval = 0
            # 保存模型的文件夹
            self.save_path = "saves"
            # 是否启动tensorboard
            self.tensorboard_on = True
            # 是否使用混合精度训练
            self.mixed_precision = True
            # 多少个子进程用于数据加载
            self.num_workers = 0

    class _Loss:
        # 损失函数
        def __init__(self):
            self.alpha = 0.25
            self.gamma = 2.0
            self.overlap_threshold = 0.5
            self.neg_pos = 3
            self.variance = [0.1, 0.2]

    class _Optimizer:
        # 优化器
        def __init__(self):
            self.name = "Adam"

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
            self.letterbox_image = True
            self.nms_threshold = 0.5
            self.confidence_threshold = 0.7
