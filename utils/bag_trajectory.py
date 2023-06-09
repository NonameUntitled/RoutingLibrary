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


class SharedBagTrajectoryMemoryQueue(BaseBagTrajectoryMemory, config_name='shared_path_memory_queue'):
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

    def get_reward(self, reward_weights: Dict[str, float]):
        return sum(r * reward_weights.get(t, 0.0) for t, r in self.reward_by_type.items())


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
    buffer_size = 10
    force_sample_length = 10
    max_trajectory_length = 15
    trajectory_by_bag_id = {}
    parts_by_node_idx = defaultdict(set)
    buffer_update_sample_count = 1
    buffer_update_sample_counter = 0

    def __init__(self, reward_weights=None):
        if reward_weights is None:
            reward_weights = {}
        self._cls = SharedBagTrajectoryMemory
        self._reward_weights = reward_weights

    @classmethod
    def create_from_config(cls, config):
        return cls(config.get('reward_weights', None))

    def add_to_trajectory(self, bag_ids: Iterable, node_idxs: Iterable, extra_infos: Iterable):
        for bag_id, node_idx, extra_info in zip(bag_ids, node_idxs, extra_infos):
            self._add_to_trajectory(int(bag_id), int(node_idx), extra_info)

    def sample_trajectories_for_node_idx(self, node_idx, count, length):
        node_idx = int(node_idx)
        for_sample_parts = [
            part for part in self._cls.parts_by_node_idx[node_idx]
            if part.reward_by_type and part.parent.for_sample
        ]
        if len(for_sample_parts) == 0:
            return []
        trajectories = [[part] for part in for_sample_parts]
        # trajectories = list(map(
        #     lambda idx: [for_sample_parts[idx]],
        #     np.random.randint(len(for_sample_parts), size=count)
        # ))
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
        return [[(part, part.get_reward(self._reward_weights)) for part in tr] for tr in trajectories]

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
