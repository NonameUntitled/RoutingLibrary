from abc import abstractmethod

from .registry import MetaParent


class BasePathMemory(metaclass=MetaParent):
    @abstractmethod
    def add_to_path(self, item, src, dst, info):
        pass

    @abstractmethod
    def get_path(self, item):
        pass


class SharedPathMemory(BasePathMemory, config_name='shared_path_memory'):
    _shared_memory = {}

    @staticmethod
    def _without_info(node):
        return {
            'node': node,
            'info': None
        }

    def add_to_path(self, item, src, dst, info):
        if item not in self._shared_memory:
            self._shared_memory[item] = [self.__class__._without_info(src)]
        assert self._shared_memory[item][-1]['node'] == src
        self._shared_memory[item][-1]['info'] = info
        self._shared_memory[item].append(self._without_info(dst))

    def get_path(self, item):
        return self._shared_memory[item]
