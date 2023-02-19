from simulation.base import BaseSimulation


class ConveyorSimulation(BaseSimulation, config_name='conveyor'):

    def run(self):
        raise NotImplementedError
