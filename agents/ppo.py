from typing import Dict, Any, Collection

import torch
from torch import Tensor

from agents import TorchAgent
from agents.base import BaseInputAdapter
from ml.encoders import BaseEncoder
from ml.ppo_encoders import BaseActor, BaseCritic
from utils.bag_trajectory import BaseBagTrajectoryMemory


class PpoInputAdapter(BaseInputAdapter, config_name='ppo_input_adapter'):
    def __init__(
            self,
            bag_id,
            node_idx,
            neighbour,
            destination,
            storage
    ):
        self._bag_id = bag_id
        self._node_idx = node_idx
        self._neighbour = neighbour
        self._destination = destination
        self._storage = storage

    @classmethod
    def create_from_config(cls, config):
        return cls(
            bag_id=config.get('bag_id', 'bag_id'),
            node_idx=config.get('node_idx', 'node_idx'),
            neighbour=config.get('neighbour', 'neighbour'),
            destination=config.get('destination', 'destination'),
            storage=config.get('storage', 'storage')
        )

    def get_input(self, inputs: Dict[str, Any]):
        return inputs[self._bag_id], \
               inputs[self._node_idx], \
               inputs[self._neighbour], \
               inputs[self._destination], \
               inputs[self._storage]

    def inputs_to_list(self, inputs: Dict[str, Any]) -> Collection:
        keys = (self._bag_id, self._node_idx, self._neighbour, self._destination, self._storage)
        return [{k: v for k, v in zip(keys, values)} for values in zip(inputs[key] for key in keys)]


class PPOAgent(TorchAgent, config_name='ppo'):
    def __init__(
            self,
            node_idx: int,
            actor: BaseActor,
            critic: BaseCritic,
            discount_factor: float,
            ratio_clip: float,
            actor_loss_weight: float,
            critic_loss_weight: float,
            bag_trajectory_memory: BaseBagTrajectoryMemory,
            ppo_input_adapter: PpoInputAdapter,
            trajectory_sample_size: int,
            trajectory_length: int
    ):
        assert 0 < discount_factor < 1, 'Incorrect `discount_factor` choice'
        assert 0 < ratio_clip < 1, 'Incorrect `ratio_clip` choice'
        super().__init__()
        self._node_idx = node_idx
        self._actor = actor
        self._critic = critic
        self._discount_factor = discount_factor
        self._ratio_clip = ratio_clip
        self._actor_loss_weight = actor_loss_weight
        self._critic_loss_weight = critic_loss_weight
        self._bag_trajectory_memory = bag_trajectory_memory
        self._ppo_input_adapter = ppo_input_adapter
        self._trajectory_sample_size = trajectory_sample_size,
        self._trajectory_length = trajectory_length

    @classmethod
    def create_from_config(cls, config):
        return cls(
            node_idx=config['node_idx'],
            actor=BaseEncoder.create_from_config(config['actor']),
            critic=BaseEncoder.create_from_config(config['critic']),
            discount_factor=config.get('discount_factor', 0.99),
            ratio_clip=config.get('ratio_clip', 0.2),
            actor_loss_weight=config.get('actor_loss_weight', 1.0),
            critic_loss_weight=config.get('critic_loss_weight', 1.0),
            bag_trajectory_memory=BaseBagTrajectoryMemory.create_from_config(config['path_memory']),
            ppo_input_adapter=BaseInputAdapter.create_from_config(config['input_adapter']),
            trajectory_sample_size=config.get('trajectory_sample_size', 100),
            trajectory_length=config.get('trajectory_length', 100)
        )

    def forward(self, inputs: Dict[str, Any]) -> Tensor:
        bag_id, node_idx, neighbour, destination, storage = self._ppo_input_adapter.get_input(inputs)
        next_node_idx, policy = self._actor.forward(
            node_idx=node_idx,
            neighbour=neighbour,
            destination=destination,
            storage=storage
        )
        v_func = self._critic.forward(
            node_idx=node_idx,
            destination=destination,
            storage=storage
        )
        self._bag_trajectory_memory.add_to_trajectory(
            bag_ids=bag_id,
            node_idxs=node_idx,
            infos=list(map(lambda v, p, inp: {
                'v_func': v,
                'policy': p,
                'inputs': inp
            }, zip(v_func, policy, self._ppo_input_adapter.inputs_to_list(inputs))))
        )
        return next_node_idx

    def learn(self):
        raise NotImplementedError

    def _loss(
            self,
            state,
            policy_old,
            start_v_old,
            rewards,
            end_v_old
    ) -> Tensor:
        _, policy = self._actor(**state)
        policy_tensor = policy
        policy_old_tensor = policy_old
        prob_ratio = policy_tensor / policy_old_tensor
        start_v_old_tensor = start_v_old

        advantage = self._compute_advantage_score(start_v_old, rewards, end_v_old)

        weighted_prob = advantage * prob_ratio
        weighted_clipped_prob = torch.clamp(prob_ratio, 1 - self.ratio_clip, 1 + self.ratio_clip) * advantage

        actor_loss = -torch.min(weighted_prob, weighted_clipped_prob)
        critic_loss = (advantage + start_v_old_tensor - self._critic(**state)) ** 2
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
