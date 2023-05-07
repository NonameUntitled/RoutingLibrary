from abc import abstractmethod
from collections import defaultdict
from collections import deque
from typing import Iterable, Dict, Any

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
            for reward_type, reward in info.reward_by_type.items():
                mean_by_type[reward_type] += reward
                count_by_type[reward_type] += 1
    for reward_type in mean_by_type:
        mean_by_type[reward_type] /= (count_by_type[reward_type] + 1e-8)
    for trajectory in trajectories:
        for info in trajectory:
            for reward_type, reward in info.reward_by_type.items():
                std_by_type[reward_type] += (reward - 0.0) ** 2
    for reward_type in std_by_type:
        std_by_type[reward_type] = (std_by_type[reward_type] / (count_by_type[reward_type] + 1e-8)) ** 0.5
    return [
        [_get_norm_reward(info.reward_by_type, mean_by_type, std_by_type) for info in trajectory]
        for trajectory in trajectories
    ]


def _get_norm_reward(reward_by_type, mean_by_type, std_by_type):
    norm_reward = 0.0
    for reward_type, reward in reward_by_type.items():
        if reward_type != 'time':
            continue
        norm_reward += (reward - 0.0) / (std_by_type[reward_type] + 1e-8)
    return norm_reward * 100


class SharedBagTrajectoryMemoryOld(BaseBagTrajectoryMemory, config_name='shared_path_memory_old'):
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


class Part:
    def __init__(
            self,
            node_idx: int,
            extra_info: Dict[str, Any],
            terminal: bool
    ):
        self.node_idx = node_idx
        self.extra_info = extra_info
        self.reward_by_type = defaultdict(float)
        self.terminal = terminal
        self.next = None
        self.parent = None

    def bind_parent(self, parent):
        self.parent = parent


class Trajectory:
    def __init__(
            self,
            bag_id: int,
            create_time: int
    ):
        self.bag_id = bag_id
        self.create_time = create_time
        self.for_sample = False
        self.root = None
        self.tail = None
        self.size = 0

    def add(self, part: Part):
        part.bind_parent(self)
        self.size += 1
        if not self.root:
            self.root = part
            self.tail = part
        else:
            self.tail.next = part
            self.tail = part

    def pop(self) -> Part:
        part = self.root
        if self.root:
            self.root = self.root.next
            self.size -= 1
        if self.size == 0:
            self.tail = None
        return part


class SharedBagTrajectoryMemory(BaseBagTrajectoryMemory, config_name='shared_path_memory'):
    trajectory_number = 0
    buffer_size = 1000
    force_sample_length = 10
    max_trajectory_length = 15
    trajectory_by_bag_id = {}
    parts_by_node_idx = defaultdict(set)
    buffer_update_sample_count = 10
    buffer_update_sample_counter = 0

    def __init__(self):
        self._cls = SharedBagTrajectoryMemory

    def add_to_trajectory(self, bag_ids: Iterable, node_idxs: Iterable, extra_infos: Iterable):
        for bag_id, node_idx, extra_info in zip(bag_ids, node_idxs, extra_infos):
            self._add_to_trajectory(int(bag_id), int(node_idx), extra_info)

    def sample_trajectories_for_node_idx(self, node_idx, count, length):
        node_idx = int(node_idx)
        trajectories = [
            [part] for part in self._cls.parts_by_node_idx[node_idx]
            if part.reward_by_type and part.parent.for_sample
        ]
        for _ in range(length - 1):
            for trajectory in trajectories:
                next_part = trajectory[-1].next
                if next_part and next_part.reward_by_type:
                    trajectory.append(next_part)
        self.buffer_update_sample_counter += 1
        if self._cls.buffer_update_sample_counter == self.buffer_update_sample_count:
            self._update_buffer()
            self._cls.buffer_update_sample_counter = 0
        if len(trajectories) == 0:
            return []
        return list(map(
            lambda idx: trajectories[idx],
            np.random.randint(len(trajectories), size=count)
        ))

    def _add_to_trajectory(self, bag_id, node_idx, extra_info):
        if bag_id not in self._cls.trajectory_by_bag_id:
            trajectory = Trajectory(bag_id, self._cls.trajectory_number)
            self._cls.trajectory_by_bag_id[bag_id] = trajectory
            self._cls.trajectory_number += 1
        part = Part(node_idx, extra_info, False)
        trajectory = self._cls.trajectory_by_bag_id[bag_id]
        self._cls.parts_by_node_idx[node_idx].add(part)
        trajectory.add(part)

    def add_reward_to_trajectory(self, bag_id, reward, reward_type, terminal=False):
        bag_id = int(bag_id)
        if bag_id not in self._cls.trajectory_by_bag_id:
            return
        trajectory = self._cls.trajectory_by_bag_id[bag_id]
        last_part = trajectory.tail
        last_part.terminal = terminal or last_part.terminal
        if last_part.terminal or trajectory.size >= self._cls.force_sample_length:
            trajectory.for_sample = True
        last_part.reward_by_type[reward_type] += reward

    def _update_buffer(self):
        complete_trajectories = sorted(
            filter(lambda tr: tr.for_sample, self.trajectory_by_bag_id.values()),
            key=lambda tr: tr.create_time
        )
        if len(complete_trajectories) > self._cls.buffer_size:
            for trajectory in complete_trajectories[self.buffer_size:]:
                for part in trajectory:
                    self._cls.parts_by_node_idx[part.node_idx].remove(part)
                del self._cls.trajectory_by_bag_id[trajectory.bag_id]
        for trajectory in self._cls.trajectory_by_bag_id.values():
            while trajectory.size > self._cls.max_trajectory_length:
                part = trajectory.pop()
                self._cls.parts_by_node_idx[part.node_idx].remove(part)
