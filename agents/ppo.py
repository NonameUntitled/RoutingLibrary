from typing import Dict, Any, Callable, Optional

import torch
from torch import Tensor, nn

from agents import TorchAgent
from ml import BaseOptimizer
from ml.encoders import BaseEncoder
from ml.ppo_encoders import BaseActor, BaseCritic
from utils.bag_trajectory import BaseBagTrajectoryMemory


class PPOAgent(TorchAgent, config_name='ppo'):

    def assign_id(self, agent_id: int):
        self._id = agent_id

    def get_id(self) -> int:
        return self._id

    def __init__(
            self,
            current_node_idx_prefix: str,
            destination_node_idx_prefix: str,
            neighbors_node_ids_prefix: str,
            output_prefix: str,
            bag_ids_prefix: str,
            actor: BaseActor,
            critic: BaseCritic,
            actor_loss_weight: float = 1.0,
            critic_loss_weight: float = 1.0,
            discount_factor: float = 0.99,
            ratio_clip: float = 0.99,
            bag_trajectory_memory: BaseBagTrajectoryMemory = None,  # Used only in training regime
            trajectory_length: int = None,  # Unused
            trajectory_sample_size: int = None,  # Used only in training regime
            optimizer_factory: Callable[[nn.Module], BaseOptimizer] = None,
    ):
        assert 0 < discount_factor < 1, 'Incorrect `discount_factor` choice'
        assert 0 < ratio_clip < 1, 'Incorrect `ratio_clip` choice'
        super().__init__()

        self._current_node_idx_prefix = current_node_idx_prefix
        self._destination_node_idx_prefix = destination_node_idx_prefix
        self._neighbors_node_ids_prefix = neighbors_node_ids_prefix
        self._output_prefix = output_prefix
        self._bag_ids_prefix = bag_ids_prefix

        self._actor = actor
        self._critic = critic

        self._actor_loss_weight = actor_loss_weight
        self._critic_loss_weight = critic_loss_weight
        self._discount_factor = discount_factor
        self._ratio_clip = ratio_clip

        self._bag_trajectory_memory = bag_trajectory_memory
        self._trajectory_length = trajectory_length
        self._trajectory_sample_size = trajectory_sample_size

        self._optimizer = optimizer_factory(self) if optimizer_factory is not None else None
        self._id = None

    @classmethod
    def create_from_config(cls, config):
        return cls(
            current_node_idx_prefix=config['current_node_idx_prefix'],
            destination_node_idx_prefix=config['destination_node_idx_prefix'],
            neighbors_node_ids_prefix=config['neighbors_node_ids_prefix'],
            output_prefix=config['output_prefix'],
            bag_ids_prefix=config.get('bag_ids_prefix', None),
            actor=BaseEncoder.create_from_config(config['actor']),
            critic=BaseEncoder.create_from_config(config['critic']),
            actor_loss_weight=config.get('actor_loss_weight', 1.0),
            critic_loss_weight=config.get('critic_loss_weight', 1.0),
            discount_factor=config.get('discount_factor', 0.99),
            ratio_clip=config.get('ratio_clip', 0.99),
            bag_trajectory_memory=BaseBagTrajectoryMemory.create_from_config(config['path_memory'])
            if 'path_memory' in config else None,
            trajectory_length=config.get('trajectory_length', 100),
            trajectory_sample_size=config.get('trajectory_sample_size', 100),
            optimizer_factory=lambda m: BaseOptimizer.create_from_config(config['optimizer'], model=m)
            if 'optimizer' in config else None,
        )

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Shape: [batch_size]
        current_node_idx = inputs[self._current_node_idx_prefix]
        batch_size = len(current_node_idx)
        # Shape: [batch_size]
        destination_node_idx = inputs[self._destination_node_idx_prefix]
        # TensorWithMask
        neighbor_node_ids = inputs[self._neighbors_node_ids_prefix]

        # Shape: [batch_size], [batch_size, max_neighbors_num]
        next_neighbors_ids, neighbors_logits = self._actor(
            current_node_idx=current_node_idx,
            neighbor_node_ids=neighbor_node_ids,
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
            bag_ids = inputs.get(self._bag_ids_prefix, None)

            self._bag_trajectory_memory.add_to_trajectory(
                bag_ids=bag_ids,
                node_idxs=torch.full([batch_size], self._id),
                extra_infos=zip(
                    torch.unsqueeze(current_state_value_function.detach(), dim=1),
                    torch.unsqueeze(current_node_idx, dim=1),
                    neighbor_node_ids,
                    torch.unsqueeze(next_neighbors_ids.detach(), dim=1),
                    torch.unsqueeze(neighbors_logits.detach(), dim=1),
                    torch.unsqueeze(destination_node_idx, dim=1)
                )
            )

        inputs[self._output_prefix] = next_neighbors_ids
        inputs.update({
            'predicted_next_node_idx': next_neighbors_ids,
            'predicted_next_node_logits': neighbors_logits,
            'predicted_current_state_v_value': current_state_value_function
        })
        return inputs

    def learn(self) -> Optional[Tensor]:
        loss = 0
        learn_trajectories = self._bag_trajectory_memory.sample_trajectories_for_node_idx(
            self._id,
            self._trajectory_sample_size
        )
        if not learn_trajectories:
            return None
        for trajectory in learn_trajectories:
            loss += self._trajectory_loss(trajectory)
        self._optimizer.step(loss)
        return loss.detach()

    def _trajectory_loss(self, trajectory):
        reward = [data.get('reward', 0) for data in trajectory]

        v_old, node_idx, neighbors, next_neighbor, neighbor_logits_old, destination = trajectory[0]['extra_info']
        _, neighbor_logits = self._actor(
            current_node_idx=node_idx,
            neighbor_node_ids=neighbors,
            destination_node_idx=destination
        )

        next_prob_old = self._get_prob(neighbors, next_neighbor, neighbor_logits_old)
        next_prob = self._get_prob(neighbors, next_neighbor, neighbor_logits)

        v = self._critic(
            current_node_idx=node_idx,
            destination_node_idx=destination
        )

        end_v_old = trajectory[-1]['extra_info'][0]
        return self._loss(next_prob, next_prob_old, v, v_old, end_v_old, reward)

    def _get_prob(self, neighbor, next_neighbor, neighbor_logits):
        neighbors_probs = torch.nn.functional.softmax(neighbor_logits, dim=1)
        neighbors_probs = neighbors_probs * neighbor.mask
        return neighbors_probs[neighbor.padded_values == torch.unsqueeze(next_neighbor, dim=1)]

    def _loss(
            self,
            next_prob,
            next_prob_old,
            v,
            v_old,
            end_v_old,
            reward,
    ) -> Tensor:
        prob_ratio = next_prob / (next_prob_old + 1e-8)

        advantage = self._compute_advantage_score(v_old, reward, end_v_old)

        weighted_prob = advantage * prob_ratio
        weighted_clipped_prob = torch.clamp(prob_ratio, 1 - self._ratio_clip, 1 + self._ratio_clip) * advantage

        actor_loss = -torch.min(weighted_prob, weighted_clipped_prob)
        critic_loss = (advantage + v_old - v) ** 2
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

