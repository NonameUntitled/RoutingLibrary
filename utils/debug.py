from topology import BaseTopology


class Debug:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(Debug, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_topology'):
            self._topology = None

    def init_topology(self, topology: BaseTopology):
        self._topology = topology

    def debug(self):
        pass
