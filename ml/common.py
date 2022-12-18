from abc import abstractmethod, ABC
from functools import partial
from typing import Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from utils import MetaParent


class BaseEncoder(metaclass=MetaParent):
    pass


class BaseDistance(BaseEncoder):
    pass


class BaseEmbedding(metaclass=MetaParent):
    """
    Abstract class for graph node embeddings.
    """

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight: str):
        raise NotImplementedError()

    def transform(self, nodes):
        raise NotImplementedError()


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


def get_optimizer(name: str, params=None):  # TODO [Vladimir Baikalov]: implement via Meta-Classes  (and enrich scheme)
    params = params or {}

    if isinstance(name, optim.Optimizer):
        return name
    if name == 'rmsprop':
        return partial(optim.RMSprop, **dict(params, lr=0.001))
    elif name == 'adam':
        return partial(optim.Adam, **params)
    elif name == 'adadelta':
        return partial(optim.Adadelta, **params)
    elif name == 'adagrad':
        return partial(optim.Adagrad, **dict(params, lr=0.001))
    else:
        raise ValueError(f'Invalid optimizer: {name}')


def add_dim(x: torch.Tensor, dim_size: int, add_first: bool) -> torch.Tensor:
    if x.dim() > dim_size:
        raise ValueError(f'Input tensor already has {x.dim()} dimensions. Required dimensions num: {dim_size}')

    while x.dim() < dim_size:
        x = x.unsqueeze(0 if add_first else -1)

    return x


class TensorWithMask:
    def __init__(self, tensor: Tensor, mask: Tensor):
        self.tensor = tensor
        self.mask = mask


class TensorLike:
    @abstractmethod
    def to_tensor(self) -> Tensor:
        pass


class Value(TensorLike, ABC):
    pass


class Reward(TensorLike, ABC):
    pass


class Policy(TensorLike, ABC):
    pass


class State(ABC):
    pass


class ThreeNodesState(State):
    def __init__(
            self, current_node: Tensor,
            neighbor_nodes: TensorWithMask,
            destination_node: Tensor
    ):
        self.current_node = current_node
        self.neighbor_nodes = neighbor_nodes
        self.destination_node = destination_node