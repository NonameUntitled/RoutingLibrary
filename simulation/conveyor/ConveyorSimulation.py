from logging import Logger

from simpy import Environment, Timeout, Event
from typing import *

from agents import TorchAgent
from simulation import BaseSimulation
from simulation.conveyor.ConveyorsEnvironment import ConveyorsEnvironment
from simulation.conveyor.utils import Bag, BagAppearanceEvent, ConveyorBreakEvent, ConveyorRestoreEvent
from topology import BaseTopology


class ConveyorSimulation(BaseSimulation, config_name='conveyor'):
    """
    Class which constructs and runs scenarios in conveyor network
    simulation environment.
    """

    def __init__(self, config: Dict[str, Any], topology: BaseTopology, agent: TorchAgent, logger: Logger):
        self._config = config
        self._world_env = Environment()
        self._topology = topology
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

        test_data = self._config['test']['data']

        bag_id = 1

        sources = [s.id for s in self._topology.sources]
        sinks = [s.id for s in self._topology.sinks]

        # for test in test_data:
        #     action = test['action']
        #     if action == 'put_bags':
        #         # delta = test['delta'] + round(np.random.normal(0, 0.5), 2)
        #         delta = test['delta']
        #
        #         cur_sources = test.get('sources', sources)
        #         cur_sinks = test.get('sinks', sinks)
        #
        #         for i in range(0, test['bags_number']):
        #             src = random.choice(cur_sources)
        #             dst = random.choice(cur_sinks)
        #
        #             mini_delta = round(abs(np.random.normal(0, 0.5)), 2)
        #             yield self._world_env.timeout(mini_delta)
        #
        #             bag = Bag(bag_id, 'sink', dst, self._world_env.now, {})
        #             yield self._simulation_env.handleEvent(BagAppearanceEvent(src, bag))
        #
        #             bag_id += 1
        #             yield self._world_env.timeout(delta)
        #     else:
        #         conv_idx = test['conv_idx']
        #         pause = test.get('pause', 0)
        #         if pause > 0:
        #             yield self._world_env.timeout(pause)
        #         if action == 'conv_break':
        #             yield self._simulation_env.handleEvent(ConveyorBreakEvent(conv_idx))
        #         else:
        #             yield self._simulation_env.handleEvent(ConveyorRestoreEvent(conv_idx))

        for i in range(0, 1500):
            bag = Bag(bag_id, 'sink', (bag_id % 10) % 3, self._world_env.now, {})
            yield self._simulation_env.handleEvent(BagAppearanceEvent(1, bag))

            bag_id += 1

            yield self._world_env.timeout(20)

    def run(self) -> None:
        """
        Runs the environment, optionally reporting the progress to a given queue.
        """
        self._world_env.process(self.runProcess())
        self._world_env.run()
