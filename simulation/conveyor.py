from typing import Tuple, List

from simulation import BaseSimulationObject
from utils import AgentId


class Conveyor(BaseSimulationObject, config_name='conveyor'):

    def __init__(self, id: str, length: float, max_speed: float, energy_consumption: float,
                 checkpoints: List[Tuple[AgentId, float]], oracle: bool):
        super().__init__(id)
        self._length = length
        self._max_speed = max_speed
        self._energy_consumption = energy_consumption
        self._checkpoints = checkpoints
        self._oracle = oracle
