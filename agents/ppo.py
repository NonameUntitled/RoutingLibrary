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
            node,
            neighbour,
            destination
    ):
        self._bag_id = bag_id
        self._node = node
        self._neighbour = neighbour
        self._destination = destination

    @classmethod
    def create_from_config(cls, config):
        return cls(
            bag_id=config.get('bag_id', 'bag_id'),
            node=config.get('node', 'node'),
            neighbour=config.get('neighbour', 'neighbour'),
            destination=config.get('destination', 'destination')
        )

    def get_input(self, inputs: Dict[str, Any]):
        return inputs[self._bag_id], inputs[self._node], inputs[self._neighbour], inputs[self._destination]

    def inputs_to_list(self, inputs: Dict[str, Any]) -> Collection:
        inputs_list = []
        for bag_id, node, neighbour, destination in zip(
                inputs[self._bag_id], inputs[self._node], inputs[self._neighbour], inputs[self._destination]
        ):
            inputs_list.append({
                self._bag_id: bag_id,
                self._node: node,
                self._neighbour: neighbour,
                self._destination: destination
            })
        return inputs_list


class PPOAgent(TorchAgent, config_name='ppo'):
    def __init__(
            self,
            actor: BaseActor,
            critic: BaseCritic,
            discount_factor: float,
            ratio_clip: float,
            actor_loss_weight: float,
            critic_loss_weight: float,
            bag_trajectory_memory: BaseBagTrajectoryMemory,
            ppo_input_adapter: PpoInputAdapter
    ):
        assert 0 < discount_factor < 1, 'Incorrect `discount_factor` choice'
        assert 0 < ratio_clip < 1, 'Incorrect `ratio_clip` choice'
        super().__init__()
        self._actor = actor
        self._critic = critic
        self._discount_factor = discount_factor
        self._ratio_clip = ratio_clip
        self._actor_loss_weight = actor_loss_weight
        self._critic_loss_weight = critic_loss_weight
        self._bag_trajectory_memory = bag_trajectory_memory
        self._ppo_input_adapter = ppo_input_adapter

    @classmethod
    def create_from_config(cls, config):
        return cls(
            actor=BaseEncoder.create_from_config(config['actor']),
            critic=BaseEncoder.create_from_config(config['critic']),
            discount_factor=config.get('discount_factor', 0.99),
            ratio_clip=config.get('ratio_clip', 0.2),
            actor_loss_weight=config.get('actor_loss_weight', 1.0),
            critic_loss_weight=config.get('critic_loss_weight', 1.0),
            bag_trajectory_memory=BaseBagTrajectoryMemory.create_from_config(config['path_memory']),
            ppo_input_adapter=BaseInputAdapter.create_from_config(config['input_adapter'])
        )

    def forward(self, inputs: Dict[str, Any]) -> Tensor:
        bag_id, node, neighbour, destination = self._ppo_input_adapter.get_input(inputs)
        next_node, policy = self._actor.forward(
            node=node,
            neighbour=neighbour,
            destination=destination,
            storage=
        )
        v_func = self._critic.forward(
            node=node,
            destination=destination,
            storage=
        )
        self._bag_trajectory_memory.add_to_trajectory(
            bag_ids=bag_id,
            nodes=node,
            infos=list(map(lambda v, p, inp: {
                'v_func': v,
                'policy': p,
                'inputs': inp
            }, zip(v_func, policy, self._ppo_input_adapter.inputs_to_list(inputs))))
        )
        return next_node

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
