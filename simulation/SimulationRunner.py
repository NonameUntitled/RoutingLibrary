from simpy import Environment, Timeout, Event
from typing import *

from simulation.ConveyorsEnvironment import ConveyorsEnvironment
from simulation.messages import Bag, BagAppearanceEvent


class SimulationRunner:
    """
    Class which constructs and runs scenarios in conveyor network
    simulation environment.
    """

    def __init__(self, run_params):
        self.config = {}
        self.env = Environment()
        self.world = ConveyorsEnvironment(run_params=run_params, env=self.env)

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
