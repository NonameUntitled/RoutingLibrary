from abc import abstractmethod

from agents import TorchAgent, State, Value


class BaseCritic(TorchAgent, config_name='base_critic'):
    @abstractmethod
    def forward(self, state: State) -> Value:
        pass
