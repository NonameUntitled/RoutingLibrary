from abc import abstractmethod, ABC
from typing import Tuple

from torch import Tensor

from agents.common import TorchAgent


# Тут можно попробовать приколов с маской накрутить
class TensorLike(ABC):
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


class BaseActor(TorchAgent, config_name='base_actor'):
    @abstractmethod
    def act(self, state: State) -> Tuple[State, Policy]:
        pass


class BaseCritic(TorchAgent, config_name='base_critic'):
    @abstractmethod
    def value(self, state: State) -> Value:
        pass
