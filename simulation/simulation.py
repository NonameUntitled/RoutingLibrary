from simulation import BaseSimulation
from simulation.conveyor import Conveyor
from simulation.diverter import Diverter


class Simulation(BaseSimulation, config_name='simulation'):

    def __init__(self, conveyors, diverters):
        super().__init__()
        self._conveyors = conveyors
        self._diverters = diverters

    @classmethod
    def create_from_config(cls, config):
        conveyors = []
        diverters = []
        for name, conf in config.items():
            if not isinstance(conf, dict):
                continue
            if conf['type'] == Conveyor.config_name:
                conveyors.append(Conveyor.create_from_config({'id': name, **conf}))
            elif conf['type'] == Diverter.config_name:
                diverters.append(Diverter.create_from_config({'id': name, **conf}))
        return cls(conveyors=conveyors, diverters=diverters)
