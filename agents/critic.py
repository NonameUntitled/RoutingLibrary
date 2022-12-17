from abc import abstractmethod

from agents.common import TorchAgent, State, Value


class BaseCritic(TorchAgent, config_name='base_critic'):
    @abstractmethod
    def forward(self, state: State) -> Value:
        pass


class TowerCritic(BaseCritic, config_name='tower_critic'):
    def __init__(self):
        super().__init__()

    def forward(self, state: State) -> Value:
        pass
