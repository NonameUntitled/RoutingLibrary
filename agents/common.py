from abc import ABC, abstractmethod

from torch import Tensor, nn

from utils import MetaParent


class TensorLike:
    @abstractmethod
    def to_tensor(self) -> Tensor:
        pass


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
