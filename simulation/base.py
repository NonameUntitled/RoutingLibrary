from utils import MetaParent


class BaseSimulation(metaclass=MetaParent):

    def run(self):
        raise NotImplementedError
