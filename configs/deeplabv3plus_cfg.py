from configs.dataset_cfg import VOC_CFG
from registry import config_registry


@config_registry("deeplabv3plus")
class DeeplabV3PlusConfig:
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
            self.backbone_name = "resnet101"
            self.backbone_pretrained = False
            # 输入图片大小
            self.input_size = (3, 513, 513)
            self.crop_size = (513, 513)
            self.output_stride = 16

    class _Dataset:
        # 数据集
        def __init__(self):
            # 目标类别数，与数据集有关，对于voc是20，对于coco是80
            self.num_classes = VOC_CFG["num_classes"] + 1
            # 数据集名称，"voc"或者"coco"
            self.dataset_name = VOC_CFG["name"]
            self.root = VOC_CFG["root"]

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
            self.warmup_iters = 0
            self.milestones = []
            self.gamma = 0.1

            # 是否使用预训练权重
            self.pretrained = False
            # 预训练模型的权重路径
            self.pretrained_weights = ""
            # 模型保存间隔
            self.save_interval = 10
            # 每隔多少epoch在验证集上验证一次
            self.eval_interval = 5
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
            self.loss_type = "focal"

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
