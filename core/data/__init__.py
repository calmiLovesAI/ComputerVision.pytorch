from configs.dataset_cfg import VOC_CFG, COCO_CFG


def find_class_name(dataset_name: str, class_index, keep_index=False):
    if dataset_name.lower() == "voc":
        class_name_list = VOC_CFG["classes"]
    else:
        class_name_list = COCO_CFG["classes"]
    if keep_index:
        return class_name_list[class_index], class_index
    return class_name_list[class_index]


def get_voc_root_and_classes():
    voc_root = VOC_CFG["root"]
    voc_classes = VOC_CFG["classes"]
    return voc_root, voc_classes