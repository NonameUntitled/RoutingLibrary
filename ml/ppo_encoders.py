from abc import abstractmethod
from typing import Tuple

from ml.common import State, Value, Policy, ThreeNodesState
from ml.encoders import TorchEncoder, TowerEncoder


class BaseActor(TorchEncoder, config_name='base_actor'):
    @abstractmethod
    def forward(self, state: State) -> Tuple[State, Policy]:
        pass


class BaseCritic(TorchEncoder, config_name='base_critic'):
    @abstractmethod
    def forward(self, state: State) -> Value:
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

    def forward(self, state: ThreeNodesState) -> Tuple[ThreeNodesState, Policy]:
        raise NotImplementedError()


class TowerCritic(BaseCritic, config_name='tower_critic'):
    def __init__(self):
        super().__init__()

    def forward(self, state: ThreeNodesState) -> Value:
        raise NotImplementedError()
