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
    def add_reward_to_trajectory(self, bag_id, reward, reward_type, terminal=False):
        raise NotImplementedError

    @abstractmethod
    def sample_trajectories_for_node_idx(self, node_idx, count, length):
        raise NotImplementedError


def get_norm_rewards(trajectories):
    mean_by_type = defaultdict(float)
    std_by_type = defaultdict(float)
    count_by_type = defaultdict(float)
    for trajectory in trajectories:
        for info in trajectory:
            for reward_type, reward in info['reward_by_type'].items():
                mean_by_type[reward_type] += reward
                count_by_type[reward_type] += 1
    for reward_type in mean_by_type:
        mean_by_type[reward_type] /= (count_by_type[reward_type] + 1e-8)
    for trajectory in trajectories:
        for info in trajectory:
            for reward_type, reward in info['reward_by_type'].items():
                std_by_type[reward_type] += (reward - 0.0) ** 2
    for reward_type in std_by_type:
        std_by_type[reward_type] = (std_by_type[reward_type] / (count_by_type[reward_type] + 1e-8)) ** 0.5
    return [
        [_get_norm_reward(info['reward_by_type'], mean_by_type, std_by_type) for info in trajectory]
        for trajectory in trajectories
    ]


def _get_norm_reward(reward_by_type, mean_by_type, std_by_type):
    norm_reward = 0.0
    for reward_type, reward in reward_by_type.items():
        if reward_type != 'time':
            continue
        norm_reward += (reward - 0.0) / (std_by_type[reward_type] + 1e-8)
    return norm_reward * 100


class SharedBagTrajectoryMemory(BaseBagTrajectoryMemory, config_name='shared_path_memory'):
    _bag_id_buffer = defaultdict(deque)
    _node_idx_buffer = defaultdict(deque)
    _buffer = deque()
    _buffer_size = 1000

    def __init__(self):
        self._cls = SharedBagTrajectoryMemory

    def add_to_trajectory(self, bag_ids: Iterable, node_idxs: Iterable, extra_infos: Iterable):
        for bag_id, node_idx, extra_info in zip(bag_ids, node_idxs, extra_infos):
            self._add_to_trajectory(int(bag_id), int(node_idx), extra_info)

    def sample_trajectories_for_node_idx(self, node_idx, count, length):
        node_idx = int(node_idx)

        all_trajectories = [[step] for step in self._cls._node_idx_buffer[node_idx] if
                            step.get('reward_by_type') is not None]
        for _ in range(length - 1):
            for trajectory in all_trajectories:
                last_step = trajectory[-1]
                if last_step.get('next') is not None and last_step['next'].get('reward_by_type') is not None:
                    trajectory.append(last_step['next'])

        # filter finished only
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

    def add_reward_to_trajectory(self, bag_id, reward, reward_type, terminal=False):
        bag_id = int(bag_id)
        if not self._cls._bag_id_buffer[bag_id]:
            return
        info = self._cls._bag_id_buffer[bag_id][-1]
        info['terminal'] = terminal or info.get('terminal', False)
        if not info.get('reward_by_type'):
            info['reward_by_type'] = defaultdict(float)
        info['reward_by_type'][reward_type] += reward
