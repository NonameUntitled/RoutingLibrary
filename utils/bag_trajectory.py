from abc import abstractmethod
from collections import defaultdict
from typing import Iterable

import numpy as np

from .registry import MetaParent


class BaseBagTrajectoryMemory(metaclass=MetaParent):
    @abstractmethod
    def add_to_trajectory(self, bag_ids: Iterable, node_idxs: Iterable, extra_infos: Iterable):
        raise NotImplementedError

    @abstractmethod
    def add_reward_to_trajectory(self, bag_id, reward):
        raise NotImplementedError

    @abstractmethod
    def sample_trajectories_for_node_idx(self, node_idx, count):
        raise NotImplementedError


class SharedBagTrajectoryMemory(BaseBagTrajectoryMemory, config_name='shared_path_memory'):
    _trajectory_by_bag_id = defaultdict(list)
    _start_trajectories_ids_by_node_idx = defaultdict(dict)  # TODO implement multiple indices support

    def add_to_trajectory(self, bag_ids: Iterable, node_idxs: Iterable, extra_infos: Iterable):
        for bag_id, node_idx, extra_info in zip(bag_ids, node_idxs, extra_infos):
            self._add_to_trajectory(bag_id, node_idx, extra_info)

    def sample_trajectories_for_node_idx(self, node_idx, count):
        data = list(self._start_trajectories_ids_by_node_idx[node_idx].items())
        return list(map(lambda idx: self._trajectory_by_bag_id[data[idx][0]][data[idx][1]:],
                        np.random.randint(len(data), size=count)))

    def _add_to_trajectory(self, bag_id, node_idx, extra_info):
        self._trajectory_by_bag_id[bag_id].append({
            'node_idx': node_idx,
            'extra_info': extra_info
        })
        self._start_trajectories_ids_by_node_idx[node_idx][bag_id] = len(self._trajectory_by_bag_id[bag_id]) - 1

    def add_reward_to_trajectory(self, bag_id, reward):
        self._trajectory_by_bag_id[bag_id][-1]['reward'] = reward
