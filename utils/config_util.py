import shutil
import yaml


def load_configs(config_path):
    with open(config_path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return configs


def save_configs(configs, save_path):
    shutil.copyfile(configs, save_path)
