from abc import abstractmethod
from typing import Any, Dict

from torch import nn

from utils import MetaParent


class BaseAgent(metaclass=MetaParent):
    pass


class TorchAgent(BaseAgent, nn.Module):
    @abstractmethod
    def forward(self, inputs: Dict[str, Any]):
        pass

    @abstractmethod
    def learn(self):
        pass

    def __init__(self):
        super().__init__()
