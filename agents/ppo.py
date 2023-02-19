from typing import Dict, Any

import torch
from torch import Tensor

from agents import TorchAgent
from ml.encoders import BaseEncoder
from ml.ppo_encoders import BaseActor, BaseCritic
from utils.bag_trajectory import BaseBagTrajectoryMemory


class PPOAgent(TorchAgent, config_name='ppo'):

    def __init__(
            self,
            current_node_idx_prefix: str,
            destination_node_idx_prefix: str,
            neighbours_node_ids_prefix: str,
            actor: BaseActor,
            critic: BaseCritic,
            actor_loss_weight: float = 1.0,
            critic_loss_weight: float = 1.0,
            discount_factor: float = 0.99,
            ratio_clip: float = 0.99,
            bag_trajectory_memory: BaseBagTrajectoryMemory = None,  # Used only in training regime
            node_idx: int = None,  # Used only in training regime
            trajectory_length: int = None,  # Unused
            trajectory_sample_size: int = None  # Used only in training regime
    ):
        assert 0 < discount_factor < 1, 'Incorrect `discount_factor` choice'
        assert 0 < ratio_clip < 1, 'Incorrect `ratio_clip` choice'
        super().__init__()

        self._current_node_idx_prefix = current_node_idx_prefix
        self._destination_node_idx_prefix = destination_node_idx_prefix
        self._neighbours_node_ids_prefix = neighbours_node_ids_prefix

        self._actor = actor
        self._critic = critic

        self._actor_loss_weight = actor_loss_weight
        self._critic_loss_weight = critic_loss_weight
        self._discount_factor = discount_factor
        self._ratio_clip = ratio_clip

        self._bag_trajectory_memory = bag_trajectory_memory
        self._node_idx = node_idx
        self._trajectory_length = trajectory_length
        self._trajectory_sample_size = trajectory_sample_size,

    @classmethod
    def create_from_config(cls, config):
        return cls(
            current_node_idx_prefix=config['current_node_idx_prefix'],
            destination_node_idx_prefix=config['destination_node_idx_prefix'],
            neighbours_node_ids_prefix=config['neighbours_node_ids_prefix'],
            actor=BaseEncoder.create_from_config(config['actor']),
            critic=BaseEncoder.create_from_config(config['critic']),
            actor_loss_weight=config.get('actor_loss_weight', 1.0),
            critic_loss_weight=config.get('critic_loss_weight', 1.0),
            discount_factor=config.get('discount_factor', 0.99),
            ratio_clip=config.get('ratio_clip', 0.99),
            bag_trajectory_memory=BaseBagTrajectoryMemory.create_from_config(config['path_memory'])
            if 'path_memory' in config else None,
            node_idx=config.get('node_idx', None),
            trajectory_length=config.get('trajectory_length', 100),
            trajectory_sample_size=config.get('trajectory_sample_size', 100)
        )

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Shape: [batch_size]
        current_node_idx = inputs[self._current_node_idx_prefix]
        # Shape: [batch_size]
        destination_node_idx = inputs[self._destination_node_idx_prefix]
        # TensorWithMask
        neighbour_node_ids = inputs[self._neighbours_node_ids_prefix]

        # Shape: [batch_size], [batch_size, max_neighbors_num]
        next_neighbors_ids, neighbors_logits = self._actor(
            current_node_idx=current_node_idx,
            neighbor_node_ids=neighbour_node_ids,
            destination_node_idx=destination_node_idx
        )

        # Shape: [batch_size]
        current_state_value_function = self._critic(
            current_node_idx=current_node_idx,
            destination_node_idx=destination_node_idx
        )

        if self._bag_trajectory_memory is not None:
            # TODO[Vladimir Baikalov]: Think about how to generalize
            # Shape: [batch_size] if exists, None otherwise
            bag_ids = inputs.get(self._bag_idx_prefix, None)

            self._bag_trajectory_memory.add_to_trajectory(
                bag_ids=bag_ids,
                node_idxs=current_node_idx,
                extra_infos=zip(
                    current_state_value_function,
                    neighbors_logits,
                    bag_ids,
                    current_node_idx,
                    neighbour_node_ids,
                    destination_node_idx
                )
            )

        return {
            'predicted_next_node_idx': next_neighbors_ids,
            'predicted_next_node_logits': neighbors_logits,
            'predicted_current_state_v_value': current_state_value_function
        }

    def learn(self):
        for trajectory in self._bag_trajectory_memory.sample_trajectories_for_node_idx(
                self._node_idx,
                self._trajectory_sample_size
        ):
            loss = self._trajectory_loss(trajectory)
            # TODO backward here? or return to env?

    def _trajectory_loss(self, trajectory):
        reward = [data['reward'] for data in trajectory]
        start_v_func, policy, bag_id, node_idx, neighbour, destination, storage = trajectory[0]['extra_infos'][0]
        end_v_func = trajectory[-1]['extra_infos'][0]
        return self._loss(node_idx, neighbour, destination, storage, policy, start_v_func, reward, end_v_func)

    def _loss(
            self,
            node_idx,
            neighbour,
            destination,
            storage,
            policy_old,
            start_v_old,
            reward,
            end_v_old
    ) -> Tensor:
        _, policy = self._actor(
            node_idx=node_idx,
            neighbour=neighbour,
            destination=destination,
            storage=storage
        )
        policy_tensor = policy
        policy_old_tensor = policy_old
        prob_ratio = policy_tensor / policy_old_tensor
        start_v_old_tensor = start_v_old

        advantage = self._compute_advantage_score(start_v_old, reward, end_v_old)

        weighted_prob = advantage * prob_ratio
        weighted_clipped_prob = torch.clamp(prob_ratio, 1 - self.ratio_clip, 1 + self.ratio_clip) * advantage

        actor_loss = -torch.min(weighted_prob, weighted_clipped_prob)
        v_func = self._critic(
            node_idx=node_idx,
            destination=destination,
            storage=storage
        )
        critic_loss = (advantage + start_v_old_tensor - v_func) ** 2
        total_loss = self._actor_loss_weight * actor_loss + self._critic_loss_weight * critic_loss

        return total_loss

    def _compute_advantage_score(
            self,
            start_v_old,
            rewards,
            end_v_old
    ):
        end_v_old_tensor = end_v_old
        advantage = start_v_old
        for reward in rewards:
            advantage = self._discount_factor * advantage + reward
        advantage -= end_v_old_tensor
        return advantage
