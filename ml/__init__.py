import torch
import torch.nn as nn

from .losses import BaseLoss
from .optimizers import BaseOptimizer
from .callbacks import BaseCallback


def get_activation(name: str):  # TODO [Vladimir Baikalov]: implement via Meta-Classes (maybe)
    if type(name) != str:
        return name
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f'Invalid activation function: {name}')


def add_dim(x: torch.Tensor, dim_size: int, add_first: bool) -> torch.Tensor:
    if x.dim() > dim_size:
        raise ValueError(f'Input tensor already has {x.dim()} dimensions. Required dimensions num: {dim_size}')

    while x.dim() < dim_size:
        x = x.unsqueeze(0 if add_first else -1)

    return x
