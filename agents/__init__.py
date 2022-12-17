from abc import ABC

from torch import nn

from utils import MetaParent
from .ppo import *


class BaseAgent(metaclass=MetaParent):
    pass


class TorchAgent(BaseAgent, nn.Module):

    def __init__(self):
        super().__init__()


class Value(TensorLike, ABC):
    pass


class Reward(TensorLike, ABC):
    pass


class Policy(TensorLike, ABC):
    pass


class State(ABC):
    pass
