from abc import abstractmethod
from typing import Any, Dict, Optional

from torch import nn, Tensor

from utils import MetaParent


class BaseAgent(metaclass=MetaParent):
    pass


class TorchAgent(BaseAgent, nn.Module):

    @abstractmethod
    def forward(self, inputs: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def learn(self) -> Optional[Tensor]:
        raise NotImplementedError

    @abstractmethod
    def assign_id(self, agent_id: int):
        raise NotImplementedError

    @abstractmethod
    def get_id(self) -> int:
        raise NotImplementedError

    def __init__(self):
        super().__init__()
