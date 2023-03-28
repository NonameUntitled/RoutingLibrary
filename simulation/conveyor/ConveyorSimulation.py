from logging import Logger

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

    def __init__(self, config: Dict[str, Any], topology: BaseTopology, agent: TorchAgent, logger: Logger):
        self._config = config
        self._world_env = Environment()
        self._simulation_env = ConveyorsEnvironment(config=self._config, world_env=self._world_env, topology=topology,
                                                    agent=agent,
                                                    logger=logger)

    @classmethod
    def create_from_config(cls, config: Dict[str, Any], topology: Optional[BaseTopology] = None,
                           agent: Optional[TorchAgent] = None, logger: Logger = None):
        assert topology is not None, "Topology must be provided"
        assert agent is not None, "Agent must be provided"
        assert logger is not None, "Logger must be provided"
        return cls(config, topology, agent, logger)

    def runProcess(self) -> Generator[Union[Timeout, Event], None, None]:
        """
        Generator which generates a series of test scenario events in
        the simulation environment.
        """

        # TODO[Aleksandr Pakulev]: Implement bugs scheduling from config

        bag_id = 1

        for i in range(0, 300):
            bag = Bag(bag_id, 'sink', 0, self._world_env.now, {})
            yield self._simulation_env.handleEvent(BagAppearanceEvent(0, bag))

            bag_id += 1

            yield self._world_env.timeout(20)

    def run(self) -> None:
        """
        Runs the environment, optionally reporting the progress to a given queue.
        """
        self._world_env.process(self.runProcess())
        self._world_env.run()
