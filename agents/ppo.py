from typing import Dict, Any

import torch
from torch import Tensor

from agents import TorchAgent
from ml.encoders import BaseEncoder

from ml.ppo_encoders import BaseActor, BaseCritic
from utils.path import BasePathMemory


class PPOAgent(TorchAgent, config_name='ppo'):
    def __init__(
            self,
            out_prefix: str,
            actor: BaseActor,
            critic: BaseCritic,
            discount_factor: float,
            ratio_clip: float,
            actor_loss_weight: float,
            critic_loss_weight: float,
            path_memory: BasePathMemory
    ):
        assert 0 < discount_factor < 1, 'Incorrect `discount_factor` choice'
        assert 0 < ratio_clip < 1, 'Incorrect `ratio_clip` choice'
        super().__init__()
        self._out_prefix = out_prefix
        self._actor = actor
        self._critic = critic
        self._discount_factor = discount_factor
        self._ratio_clip = ratio_clip

        self._actor_loss_weight = actor_loss_weight
        self._critic_loss_weight = critic_loss_weight
        self._path_memory = path_memory

    @classmethod
    def create_from_config(cls, config):
        return cls(
            out_prefix=config.get('out_prefix', config['prefix']),
            actor=BaseEncoder.create_from_config(config['actor']),
            critic=BaseEncoder.create_from_config(config['critic']),
            discount_factor=config.get('discount_factor', 0.99),
            ratio_clip=config.get('ratio_clip', 0.2),
            actor_loss_weight=config.get('actor_loss_weight', 1.0),
            critic_loss_weight=config.get('critic_loss_weight', 1.0),
            path_memory=BasePathMemory.create_from_config(config['path_memory'])
        )

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ppo_input = inputs[self._prefix]
        next_state, policy = self._actor.forward(**ppo_input)
        curr_v_func = self._critic.forward(**ppo_input)  # (batch_size)
        q_estimate = self._critic.forward(**{**ppo_input, 'state': next_state})  # (batch_size)
        self._memory.append({
            'state': ppo_input,
            'policy': policy,
            'curr_v_func': curr_v_func,
            'q_estimate': q_estimate
        })
        inputs[self._out_prefix] = next_state
        return inputs

    def learn(self):
        pass

    def _loss(
            self,
            state,
            policy_old,
            start_v_old,
            # Если есть трейсы разной длины видимо надо с маской что-то делать
            rewards,
            end_v_old
    ) -> Tensor:
        _, policy = self._actor(state)
        policy_tensor = policy
        policy_old_tensor = policy_old
        prob_ratio = policy_tensor / policy_old_tensor
        start_v_old_tensor = start_v_old

        advantage = self._compute_advantage_score(start_v_old, rewards, end_v_old)

        weighted_prob = advantage * prob_ratio
        weighted_clipped_prob = torch.clamp(prob_ratio, 1 - self.ratio_clip, 1 + self.ratio_clip) * advantage

        actor_loss = -torch.min(weighted_prob, weighted_clipped_prob)
        critic_loss = (advantage + start_v_old_tensor - self._critic(state)) ** 2
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
