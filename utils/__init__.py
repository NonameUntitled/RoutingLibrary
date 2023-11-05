import argparse
import json
import logging
import os

import numpy as np
import random
import torch

from .registry import MetaParent
from .tensorboard_writers import TensorboardTimer, TensorboardWriter

CHECKPOINT_DIR = './checkpoints'
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE = torch.device('cpu')


def create_logger(
        name,
        level=logging.DEBUG,
        format='[%(asctime)s] [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
):
    logging.basicConfig(level=level, format=format, datefmt=datefmt)
    logger = logging.getLogger(name)
    return logger


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)


def maybe_to_list(values):
    if not isinstance(values, list):
        values = [values]
    return values


def read_json_file(file_path):
    with open(file_path) as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True)
    args = parser.parse_args()
    params = read_json_file(args.params)
    if 'topology' in params and isinstance(params['topology'], str):
        topology_path = params['topology']
        if not os.path.exists(topology_path):
            topology_path = os.path.join(os.path.dirname(args.params), topology_path)
        if os.path.exists(topology_path):
            topology_data = read_json_file(topology_path)
            params['topology'] = topology_data['topology']
        else:
            raise FileNotFoundError(f"Could not find the topology file at {topology_path}")
    return params


def shared(original_class):
    cls_create_from_config = original_class.create_from_config

    original_class._SHARED_INSTANCE = None

    def create_from_config(config):
        use_shared = config.get('shared', False)
        if use_shared:
            if original_class._SHARED_INSTANCE is None:
                # Create a new one
                original_class._SHARED_INSTANCE = cls_create_from_config(config)
            else:
                # Use a shares one
                pass
            return original_class._SHARED_INSTANCE
        else:
            return cls_create_from_config(config)

    original_class.create_from_config = create_from_config
    return original_class
