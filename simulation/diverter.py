from simulation import BaseSimulationObject


class Diverter(BaseSimulationObject, config_name='diverter'):
    def __init__(self, id):
        super().__init__(id)
