from abc import abstractmethod
from collections import defaultdict
from typing import Collection

import numpy as np

from .registry import MetaParent


class BaseBagTrajectoryMemory(metaclass=MetaParent):
    @abstractmethod
    def add_to_trajectory(self, bag_ids: Collection, nodes: Collection, infos: Collection):
        raise NotImplementedError

    @abstractmethod
    def add_reward_to_trajectory(self, bag_id, reward):
        raise NotImplementedError

    @abstractmethod
    def sample_trajectories_for_node(self, node, count):
        raise NotImplementedError


class SharedBagTrajectoryMemory(BaseBagTrajectoryMemory, config_name='shared_path_memory'):
    _trajectory_by_bag_id = defaultdict(list)
    _start_trajectories_ids_by_node = defaultdict(dict)  # TODO implement multiple indices support

    def add_to_trajectory(self, bag_ids: Collection, nodes: Collection, infos: Collection):
        for bag_id, node, info in zip(bag_ids, nodes, infos):
            self._add_to_trajectory(bag_id, node, info)

    def sample_trajectories_for_node(self, node, count):
        data = list(self._start_trajectories_ids_by_node[node].items())
        return list(map(lambda idx: self._trajectory_by_bag_id[data[idx][0]][data[idx][1]:],
                        np.random.randint(len(data), size=count)))

    def _add_to_trajectory(self, bag_id, node, info):
        self._trajectory_by_bag_id[bag_id].append({
            'node': node,
            'info': info
        })
        self._start_trajectories_ids_by_node[node][bag_id] = len(self._trajectory_by_bag_id[bag_id]) - 1

    def add_reward_to_trajectory(self, bag_id, reward):
        self._trajectory_by_bag_id[bag_id][-1]['info']['reward'] = reward
