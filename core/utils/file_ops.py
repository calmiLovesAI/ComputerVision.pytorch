import os
import torch


def get_root_absolute_path():
    # 获取当前文件所在的绝对路径
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # 获取上两级目录的绝对路径
    return os.path.abspath(os.path.join(current_directory, '..', '..'))


def auto_make_dirs(file_path):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def download_file(url, model_dir):
    root_absolute_path = get_root_absolute_path()
    model_dir = os.path.join(root_absolute_path, model_dir)
    auto_make_dirs(model_dir)
    if os.path.exists(model_dir):
        print(f"File '{model_dir}' already exists.")
    else:
        print(f"Start downloading from: {url}")
        torch.hub.download_url_to_file(url=url, dst=model_dir)
        # r = requests.get(url, stream=True)
        # total = int(r.headers.get("content-length", 0))
        # with open(model_dir, mode="wb") as f, \
        #         tqdm(desc=model_path, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
        #     for data in r.iter_content(chunk_size=1024):
        #         size = f.write(data)
        #         bar.update(size)
        print("Download completed!")


def load_state_dict_from_url(url, model_dir, map_location=None):
    download_file(url, model_dir)
    return torch.load(model_dir, map_location=map_location)