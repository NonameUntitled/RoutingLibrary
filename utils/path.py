from abc import abstractmethod
from collections import defaultdict
from typing import Collection

from .registry import MetaParent


class BasePathMemory(metaclass=MetaParent):
    @abstractmethod
    def add_to_path(self, items: Collection, srcs: Collection, dsts: Collection, infos: Collection):
        raise NotImplementedError

    @abstractmethod
    def get_path(self, item):
        raise NotImplementedError

    @abstractmethod
    def get_paths_for_node(self, node):
        raise NotImplementedError


class SharedPathMemory(BasePathMemory, config_name='shared_path_memory'):
    _shared_memory = defaultdict(list)
    _path_by_node = defaultdict(set)

    @staticmethod
    def _without_info(node):
        return {
            'node': node,
            'info': None
        }

    def add_to_path(self, items: Collection, srcs: Collection, dsts: Collection, infos: Collection):
        for item, src, dst, info in zip(items, srcs, dsts, infos):
            self._add_to_path(item, src, dst, info)

    def get_path(self, item):
        return self._shared_memory[item]

    def get_paths_for_node(self, node):
        return self._path_by_node[node]

    def _add_to_path(self, item, src, dst, info):
        if item not in self._shared_memory:
            self._shared_memory[item] = [self.__class__._without_info(src)]
        assert self._shared_memory[item][-1]['node'] == src
        self._shared_memory[item][-1]['info'] = info
        self._shared_memory[item].append(self._without_info(dst))
        if self._shared_memory[item] not in self._path_by_node[dst]:
            self._path_by_node[dst].add(self._shared_memory[item])
