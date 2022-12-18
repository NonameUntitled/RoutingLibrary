from abc import abstractmethod
from typing import Tuple

from agents.common import TorchAgent, State, Policy, ThreeNodesState
from ml.encoders import TowerEncoder


class BaseActor(TorchAgent, config_name='base_actor'):
    @abstractmethod
    def forward(self, state: State) -> Tuple[State, Policy]:
        pass


class TowerActor(BaseActor, config_name='tower_actor'):
    def __init__(self, ff_net):
        super().__init__()
        self._ff_net = ff_net

    @classmethod
    def create_from_config(cls, config):
        return cls(
            ff_net=TowerEncoder.create_from_config(config['ff_net'])
        )

    def forward(self, state: ThreeNodesState) -> Tuple[State, Policy]:
        raise NotImplementedError()
