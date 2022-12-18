import argparse
import json
import logging
import os

from .registry import MetaParent
from .tensorboard_writers import TensorboardTimer, TensorboardWriter

GLOBAL_TENSORBOARD_WRITER = None
LOGS_DIR = './tensorboard_logs'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True)
    args = parser.parse_args()
    with open(args.params) as f:
        params = json.load(f)

    if 'experiment_name' in params:
        global GLOBAL_TENSORBOARD_WRITER
        GLOBAL_TENSORBOARD_WRITER = create_tensorboard(params['experiment_name'])

    return params


def create_tensorboard(experiment_name):
    return TensorboardWriter(log_dir=os.path.join(LOGS_DIR, experiment_name))


def create_logger(
        name,
        level=logging.DEBUG,
        format='[%(asctime)s] [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
):
    logging.basicConfig(level=level, format=format, datefmt=datefmt)
    logger = logging.getLogger(name)
    return logger


def maybe_to_list(values):
    if not isinstance(values, list):
        values = [values]
    return values
