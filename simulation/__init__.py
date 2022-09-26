from utils.registry import MetaParent


class BaseSimulation(metaclass=MetaParent):
    pass


class BaseSimulationObject(metaclass=MetaParent):
    def __init__(self, id: str):
        self.id = id
