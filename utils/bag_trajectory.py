from abc import abstractmethod
from collections import defaultdict
from collections import deque
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
    def finish_trajectory(self, bag_id):
        raise NotImplementedError

    @abstractmethod
    def sample_trajectories_for_node_idx(self, node_idx, count):
        raise NotImplementedError


class SharedBagTrajectoryMemory(BaseBagTrajectoryMemory, config_name='shared_path_memory'):
    _trajectory_by_bag_id = defaultdict(list)
    _idxs_in_trajectory_by_node_idx = defaultdict(lambda: defaultdict(list))
    _sinked_bag_ids = set()
    _to_delete_queue = deque()
    _buffer_size = 100

    def __init__(self, buffer_size: int = 100):
        self._cls = SharedBagTrajectoryMemory
        self._cls._buffer_size = buffer_size

    @classmethod
    def create_from_config(cls, config):
        return cls(
            buffer_size=config.get('buffer_size', 100)
        )

    def add_to_trajectory(self, bag_ids: Iterable, node_idxs: Iterable, extra_infos: Iterable):
        for bag_id, node_idx, extra_info in zip(bag_ids, node_idxs, extra_infos):
            self._add_to_trajectory(int(bag_id), int(node_idx), extra_info)

    def sample_trajectories_for_node_idx(self, node_idx, count):
        node_idx = int(node_idx)
        all_node_trajectories = [
            self._cls._trajectory_by_bag_id[bag_id][start_idx:]
            for bag_id, starts in filter(
                lambda bag_id_start: bag_id_start[0] in self._cls._sinked_bag_ids,
                (
                    (bag_id, starts)
                    for bag_id, starts in self._cls._idxs_in_trajectory_by_node_idx[node_idx].items()
                )
            )
            for start_idx in starts
        ]
        if len(all_node_trajectories) == 0:
            return []
        return list(map(lambda idx: all_node_trajectories[idx],
                        np.random.randint(len(all_node_trajectories), size=count)))

    def _add_to_trajectory(self, bag_id, node_idx, extra_info):
        self._cls._trajectory_by_bag_id[bag_id].append({
            'node_idx': node_idx,
            'extra_info': extra_info
        })
        self._cls._idxs_in_trajectory_by_node_idx[node_idx][bag_id].append(
            len(self._cls._trajectory_by_bag_id[bag_id]) - 1
        )

    def add_reward_to_trajectory(self, bag_id, reward):
        bag_id = int(bag_id)
        if bag_id not in self._cls._trajectory_by_bag_id:
            return
        self._cls._trajectory_by_bag_id[bag_id][-1]['reward'] = reward
        self._cls._to_delete_queue.appendleft(bag_id)
        if len(self._cls._to_delete_queue) > self._cls._buffer_size:
            bag_to_delete = self._cls._to_delete_queue.pop()
            self._delete_bag(bag_to_delete)
        return

    def finish_trajectory(self, bag_id):
        bag_id = int(bag_id)
        if bag_id not in self._cls._trajectory_by_bag_id:
            return
        self._cls._sinked_bag_ids.add(bag_id)

    def _delete_bag(self, bag_id):
        del self._cls._trajectory_by_bag_id[bag_id]
        if bag_id in self._cls._sinked_bag_ids:
            self._cls._sinked_bag_ids.remove(bag_id)
        for starts_by_bag_id in self._cls._idxs_in_trajectory_by_node_idx.values():
            if bag_id in starts_by_bag_id:
                del starts_by_bag_id[bag_id]
