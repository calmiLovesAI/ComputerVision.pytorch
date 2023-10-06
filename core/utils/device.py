import torch


def get_currently_accessible_device() -> torch.device:
    """
    获取当前可访问的设备，自动设置为显存容量最大的GPU，如果没有GPU则设置为CPU
    :return:
    """
    if torch.cuda.is_available():
        # 获取所有可用的GPU设备数量
        num_gpus = torch.cuda.device_count()

        if num_gpus == 1:
            return torch.device("cuda:0")  # 如果只有一个GPU，返回它

        # 获取每个GPU的显存容量
        gpu_memory = [torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)]

        # 找到显存容量最大的GPU的索引
        max_memory_index = gpu_memory.index(max(gpu_memory))

        return torch.device(f"cuda:{max_memory_index}")
    else:
        return torch.device("cpu")  # 如果没有可用的GPU，返回CPU设备
