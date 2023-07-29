from typing import Dict, Any, Callable, Optional

import torch
from torch import Tensor, nn

from agents import TorchAgent
from ml import BaseOptimizer
from ml.dqn_encoders import BaseQNetwork
from ml.encoders import BaseEncoder
from utils.bag_trajectory import BaseBagTrajectoryMemory


class ReinforceAgent(TorchAgent, config_name='reinforce'):
    def __init__(
            self,
            current_node_idx_prefix: str,
            destination_node_idx_prefix: str,
            neighbors_node_ids_prefix: str,
            output_prefix: str,
            bag_ids_prefix: str,
            q_network: BaseQNetwork,
            discount_factor: float = 0.99,
            bag_trajectory_memory: BaseBagTrajectoryMemory = None,
            trajectory_sample_size: int = 30,
            optimizer_factory: Callable[[nn.Module], BaseOptimizer] = None
    ):
        assert 0 < discount_factor < 1, 'Incorrect `discount_factor` choice'
        super().__init__()

        self._current_node_idx_prefix = current_node_idx_prefix
        self._destination_node_idx_prefix = destination_node_idx_prefix
        self._neighbors_node_ids_prefix = neighbors_node_ids_prefix
        self._output_prefix = output_prefix
        self._bag_ids_prefix = bag_ids_prefix

        self._q_network = q_network
        self._discount_factor = discount_factor
        self._bag_trajectory_memory = bag_trajectory_memory
        self._trajectory_sample_size = trajectory_sample_size

        self._optimizer = optimizer_factory(self) if optimizer_factory is not None else None

    @classmethod
    def create_from_config(cls, config):
        return cls(
            current_node_idx_prefix=config['current_node_idx_prefix'],
            destination_node_idx_prefix=config['destination_node_idx_prefix'],
            neighbors_node_ids_prefix=config['neighbors_node_ids_prefix'],
            output_prefix=config['output_prefix'],
            bag_ids_prefix=config.get('bag_ids_prefix', None),
            q_network=BaseEncoder.create_from_config(config['q_network']),
            discount_factor=config.get('discount_factor', 0.99),
            bag_trajectory_memory=BaseBagTrajectoryMemory.create_from_config(config['path_memory'])
            if 'path_memory' in config else None,
            trajectory_sample_size=config.get('trajectory_sample_size', 30),
            optimizer_factory=lambda m: BaseOptimizer.create_from_config(config['optimizer'], model=m)
            if 'optimizer' in config else None
        )

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Shape: [batch_size]
        current_node_idx = inputs[self._current_node_idx_prefix]
        batch_size = len(current_node_idx)
        # Shape: [batch_size]
        destination_node_idx = inputs[self._destination_node_idx_prefix]
        # Shape: [batch_size, max_neighbors_num]
        neighbor_node_ids = inputs[self._neighbors_node_ids_prefix]

        # Shape: [batch_size], [batch_size]
        next_neighbor_ids, next_neighbor_log_prob, _ = self._q_network(
            current_node_idx=current_node_idx,
            neighbor_node_ids=neighbor_node_ids,
            destination_node_idx=destination_node_idx
        )

        if self._bag_trajectory_memory is not None:
            # TODO[Vladimir Baikalov]: Think about how to generalize
            bag_ids = inputs.get(self._bag_ids_prefix, None)

            if bag_ids is not None:
                self._bag_trajectory_memory.add_to_trajectory(
                    bag_ids=bag_ids,
                    node_idxs=torch.full([batch_size], self._node_id),
                    extra_infos=next_neighbor_log_prob
                )

        inputs[self._output_prefix] = next_neighbor_ids
        return inputs

    def learn(self) -> Optional[Tensor]:
        # TODO[Vladimir Baikalov] check
        loss = 0
        learn_trajectories = self._bag_trajectory_memory.sample_trajectories_for_node_idx(
            node_idx=self._node_id,
            count=self._trajectory_sample_size,
            length=10
        )
        if not learn_trajectories:
            return None
        for trajectory in learn_trajectories[:1]:
            next_neighbor_log_prob = trajectory[0][0].extra_info

            total_reward = 0.0
            for _, reward in trajectory[::-1]:
                total_reward = reward + self._discount_factor * total_reward
            loss += -next_neighbor_log_prob * total_reward

        loss /= len(learn_trajectories)
        self._optimizer.step(loss)
        return loss.detach().item()
