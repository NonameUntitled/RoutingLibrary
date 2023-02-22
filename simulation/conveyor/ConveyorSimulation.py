from simpy import Environment, Timeout, Event
from typing import *

from simulation import BaseSimulation
from simulation.conveyor.ConveyorsEnvironment import ConveyorsEnvironment
from simulation.messages import Bag, BagAppearanceEvent
from topology import BaseTopology


class ConveyorSimulation(BaseSimulation, config_name='conveyor'):
    """
    Class which constructs and runs scenarios in conveyor network
    simulation environment.
    """

    def __init__(self, config, topology, agent):
        self.config = config
        self.env = Environment()
        self.world = ConveyorsEnvironment(config=config, env=self.env, topology=topology, agent=agent)

    @classmethod
    def create_from_config(cls, config, topology=None, agent=None):
        return cls(config, topology, agent)

    def runProcess(self) -> Generator[Timeout | Event, None, None]:
        """
        Generator which generates a series of test scenario events in
        the world environment.
        """

        bag_id = 1

        for i in range(0, 10):
            bag = Bag(bag_id, ('sink', 0), self.env.now, None)
            yield self.world.handleEvent(BagAppearanceEvent(0, bag))

            bag_id += 1

            yield self.env.timeout(20)

    def run(self) -> None:
        """
        Runs the environment, optionally reporting the progress to a given queue.
        """
        self.env.process(self.runProcess())
        self.env.run()
