import yaml


def load_yaml(filepath):
    """
    将.yaml文件解析为python字典
    :param filepath:
    :return:
    """
    with open(file=filepath, encoding='utf-8') as f:
        parsed_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    return parsed_dict


