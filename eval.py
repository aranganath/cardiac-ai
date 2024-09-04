import argparse
import os
import sys
import tempfile
import time

from engine import Engine
from utils import config_util
from utils.general_util import log, get_temp_dir, mkdir_p


def main(args):
    assert os.path.exists(args.save_dir)

    if not args.config_path:
        configs = config_util.load_configs(os.path.join(args.save_dir, "configs.yaml"))
    else:
        configs = config_util.load_configs(args.config_path)

    engine = Engine(mode='test', configs=configs, save_dir=args.save_dir)
    engine.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='',
                        help="path to a config")
    parser.add_argument('--save_dir', default='',
                        help="Previous training directory to perform evaluation")
    args = parser.parse_args()

    main(args)