from abc import abstractmethod
from typing import Tuple

from agents.common import TorchAgent, State, Policy


class BaseActor(TorchAgent, config_name='base_actor'):
    @abstractmethod
    def forward(self, state: State) -> Tuple[State, Policy]:
        pass


class TowerActor(BaseActor, config_name='tower_actor'):
    def __init__(self):
        super().__init__()

    def forward(self, state: State) -> Tuple[State, Policy]:
        pass
