import argparse
import json
import logging
import numpy as np
import os
import random
import torch

from .registry import MetaParent
from .tensorboard_writers import TensorboardTimer, TensorboardWriter

CHECKPOINT_DIR = '../checkpoints'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
GLOBAL_TENSORBOARD_WRITER = None
LOGS_DIR = '../tensorboard_logs'


def create_logger(
        name,
        level=logging.DEBUG,
        format='[%(asctime)s] [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
):
    logging.basicConfig(level=level, format=format, datefmt=datefmt)
    logger = logging.getLogger(name)
    return logger


def create_tensorboard(experiment_name):
    return TensorboardWriter(log_dir=os.path.join(LOGS_DIR, experiment_name))


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_to_list(values):
    if not isinstance(values, list):
        values = [values]
    return values

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True)
    args = parser.parse_args()
    with open(args.params) as f:
        params = json.load(f)
    return params
