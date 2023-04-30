import copy
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

    def __init__(self):
        super().__init__()
        self._node_id = None

    def copy(self, node_id: int):
        agent_copy = copy.deepcopy(self)
        agent_copy._node_id = node_id
