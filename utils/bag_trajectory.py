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
    def add_reward_to_trajectory(self, bag_id, reward, terminal=False):
        raise NotImplementedError

    @abstractmethod
    def sample_trajectories_for_node_idx(self, node_idx, count, length):
        raise NotImplementedError


class SharedBagTrajectoryMemory(BaseBagTrajectoryMemory, config_name='shared_path_memory'):
    _bag_id_buffer = defaultdict(deque)
    _node_idx_buffer = defaultdict(deque)
    _buffer = deque()
    _buffer_size = 10000

    def __init__(self):
        self._cls = SharedBagTrajectoryMemory

    def add_to_trajectory(self, bag_ids: Iterable, node_idxs: Iterable, extra_infos: Iterable):
        for bag_id, node_idx, extra_info in zip(bag_ids, node_idxs, extra_infos):
            self._add_to_trajectory(int(bag_id), int(node_idx), extra_info)

    def sample_trajectories_for_node_idx(self, node_idx, count, length):
        node_idx = int(node_idx)

        all_trajectories = [[step] for step in self._cls._node_idx_buffer[node_idx] if step.get('reward') is not None]
        for _ in range(length - 1):
            for trajectory in all_trajectories:
                last_step = trajectory[-1]
                if last_step.get('next') is not None and last_step['next'].get('reward') is not None:
                    trajectory.append(last_step['next'])

        all_trajectories = list(filter(lambda tr: tr[-1]['terminal'], all_trajectories))

        if len(all_trajectories) == 0:
            return []
        return list(map(lambda idx: all_trajectories[idx],
                        np.random.randint(len(all_trajectories), size=count)))

    def _add_to_trajectory(self, bag_id, node_idx, extra_info):
        step = {
            'bag_id': bag_id,
            'node_idx': node_idx,
            'extra_info': extra_info,
        }
        if len(self._cls._bag_id_buffer[bag_id]) > 0:
            prev = self._cls._bag_id_buffer[bag_id][-1]
            prev['next'] = step
        self._cls._buffer.append(step)
        self._cls._bag_id_buffer[bag_id].append(step)
        self._cls._node_idx_buffer[node_idx].append(step)
        while len(self._cls._buffer) > self._cls._buffer_size:
            step = self._cls._buffer.popleft()
            self._cls._bag_id_buffer[step['bag_id']].popleft()
            self._cls._node_idx_buffer[step['node_idx']].popleft()

    def add_reward_to_trajectory(self, bag_id, reward, terminal=False):
        bag_id = int(bag_id)
        if bag_id not in self._cls._bag_id_buffer:
            return
        info = self._cls._bag_id_buffer[bag_id][-1]
        info['terminal'] = terminal or info.get('terminal', False)
        if 'reward' not in info:
            info['reward'] = 0
        info['reward'] += reward
