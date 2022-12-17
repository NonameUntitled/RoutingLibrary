from abc import abstractmethod
from typing import Tuple

from agents import TorchAgent, State, Policy


class BaseActor(TorchAgent, config_name='base_actor'):
    @abstractmethod
    def forward(self, state: State) -> Tuple[State, Policy]:
        pass
