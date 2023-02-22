from simpy import Environment, Timeout, Event
from typing import *

from agents import TorchAgent
from simulation import BaseSimulation
from simulation.conveyor.ConveyorsEnvironment import ConveyorsEnvironment
from simulation.conveyor.utils import Bag, BagAppearanceEvent
from topology import BaseTopology


class ConveyorSimulation(BaseSimulation, config_name='conveyor'):
    """
    Class which constructs and runs scenarios in conveyor network
    simulation environment.
    """

    def __init__(self, config: Dict[str, Any], topology: BaseTopology, agent: TorchAgent):
        self._config = config
        self._env = Environment()
        self._world = ConveyorsEnvironment(config=self._config, env=self._env, topology=topology, agent=agent)

    @classmethod
    def create_from_config(cls, config: Dict[str, Any], topology: Optional[BaseTopology] = None, agent: Optional[TorchAgent] = None):
        assert topology is not None, "Topology must be provided"
        assert agent is not None, "Agent must be provided"
        return cls(config, topology, agent)

    def runProcess(self) -> Generator[Timeout | Event, None, None]:
        """
        Generator which generates a series of test scenario events in
        the world environment.
        """

        bag_id = 1

        for i in range(0, 10):
            bag = Bag(bag_id, 'sink', 0, self._env.now, {})
            yield self._world.handleEvent(BagAppearanceEvent(0, bag))

            bag_id += 1

            yield self._env.timeout(20)

    def run(self) -> None:
        """
        Runs the environment, optionally reporting the progress to a given queue.
        """
        self._env.process(self.runProcess())
        self._env.run()
