
class Config:
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
            # 目标类别数，与数据集有关，对于voc是20，对于coco是80
            self.num_classes = 20
            # 输入图片大小：(C, H, W)
            self.input_size = (3, 416, 416)
            # [w, h]
            self.anchor = [116, 90, 156, 198, 373, 326, 30, 61, 62, 45, 59, 119, 10, 13, 16, 30, 33, 23]
            self.anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
            self.output_features = [13, 26, 52]

    class _Dataset:
        # 数据集
        def __init__(self):
            # 数据集名称，"voc"或者"coco"
            self.dataset_name = "voc"

    class _Train:
        # 训练参数
        def __init__(self):
            # 恢复训练时加载的checkpoint文件，""表示从epoch=0开始训练
            # 测试时也需要在这里指定checkpoint文件
            self.resume_training = "saves/YoloV3_voc_epoch-195.pth"
            # 恢复训练时的上一次epoch是多少，-1表示从epoch=0开始训练
            self.last_epoch = 195

            self.epoch = 250
            self.batch_size = 8
            # 初始学习率
            self.initial_lr = 1e-3
            # warm up轮数
            self.warmup_iters = 0
            self.milestones = [197, 225]
            self.gamma = 0.1

            # 是否使用预训练权重
            self.pretrained = False
            # 模型保存间隔
            self.save_interval = 5
            # 每隔多少epoch在验证集上验证一次
            self.eval_interval = 1
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

    class _Optimizer:
        # 优化器
        def __init__(self):
            self.name = "Adam"

    class _Log:
        # 训练日志
        def __init__(self):
            # 日志文件保存文件夹
            self.root = "out"
            # 日志文件输出间隔
            self.print_interval = 50

    class _Decode:
        def __init__(self):
            self.test_results = "result"
            self.max_boxes_per_img = 100
            self.conf_threshold = 0.6
            self.iou_threshold = 0.5
