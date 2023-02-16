from typing import Dict, Any

from torch import Tensor

from agents import TorchAgent
from ml.dqn_encoders import BaseQNetwork
from ml.encoders import BaseEncoder
from utils.bag_trajectory import BaseBagTrajectoryMemory


class DQNAgent(TorchAgent, config_name='dqn'):
    def __init__(
            self,
            node_idx: int,
            q_network: BaseQNetwork,
            discount_factor: float,
            research_prob: float,
            bag_trajectory_memory: BaseBagTrajectoryMemory,
            sample_size: int,
    ):
        assert 0 < discount_factor < 1, 'Incorrect `discount_factor` choice'
        assert 0 < research_prob < 1, 'Incorrect `discount_factor` choice'
        super().__init__()
        self._node_idx = node_idx
        self._q_network = q_network
        self._discount_factor = discount_factor
        self._research_prob = research_prob
        self._bag_trajectory_memory = bag_trajectory_memory
        self._sample_size = sample_size

    @classmethod
    def create_from_config(cls, config):
        return cls(
            node_idx=config['node_idx'],
            q_network=BaseEncoder.create_from_config(config['q_network']),
            discount_factor=config.get('discount_factor', 0.99),
            research_prob=config.get('research_prob', 0.1),
            bag_trajectory_memory=BaseBagTrajectoryMemory.create_from_config(config['path_memory']),
            sample_size=config.get('sample_size', 100)
        )

    def forward(self, inputs: Dict[str, Any]) -> Tensor:
        # Shape: [batch_size]
        current_node_idx = inputs[self._current_node_idx_prefix]
        # Shape: [batch_size]
        destination_node_idx = inputs[self._destination_node_idx_prefix]
        # TensorWithMask
        neighbour_node_ids = inputs[self._neighbours_node_ids_prefix]

        next_neighbour, _ = self._q_network(
            current_node_idx=current_node_idx,
            neighbor_node_ids=neighbour_node_ids,
            destination_node_idx=destination_node_idx,
            research_prob=self._research_prob
        )

        if self._bag_trajectory_memory is not None:
            # TODO[Vladimir Baikalov]: Think about how to generalize
            bag_ids = inputs.get(self._bag_idx_prefix, None)

            self._bag_trajectory_memory.add_to_trajectory(
                bag_ids=bag_ids,
                node_idxs=current_node_idx,
                extra_infos=zip(
                    current_node_idx,
                    neighbour_node_ids,
                    destination_node_idx,
                    next_neighbour
                )
            )
        return next_neighbour

    def learn(self):
        for trajectory in self._bag_trajectory_memory.sample_trajectories_for_node_idx(
                self._node_idx,
                self._trajectory_sample_size
        ):
            target = trajectory[0]['reward']
            current_node_idx, neighbour_node_ids, destination_node_idx, next_neighbour = trajectory[0]['extra_infos']
            if len(trajectory) > 0:
                current_node_idx_, neighbour_node_ids_, destination_node_idx_, _ = trajectory[1]['extra_infos']
                _, neighbour_q_ = self._q_network(
                    current_node_idx=current_node_idx_,
                    neighbor_node_ids=neighbour_node_ids_,
                    destination_node_idx=destination_node_idx_,
                    research_prob=self._research_prob
                )
                target += self._discount_factor * neighbour_q_
            _, neighbour_q = self._q_network(
                current_node_idx=current_node_idx,
                neighbor_node_ids=neighbour_node_ids,
                destination_node_idx=destination_node_idx,
                research_prob=self._research_prob
            )
            loss = (target - neighbour_q[next_neighbour]) ** 2
