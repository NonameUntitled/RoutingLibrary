from typing import List, Tuple

import torch
from torch import Tensor

from agents import TorchAgent

from ml.encoders import BaseEncoder
from ml.ppo_encoders import BaseActor, BaseCritic
from ml.typing import Reward, Value, Policy, State


class PPOAgent(TorchAgent, config_name='ppo'):
    def __init__(
            self,
            actor: BaseActor,
            critic: BaseCritic,
            discount_factor: float,
            ratio_clip: float,
            actor_loss_weight: float,
            critic_loss_weight: float
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

    @classmethod
    def create_from_config(cls, config):
        return cls(
            actor=BaseEncoder.create_from_config(config['actor']),
            critic=BaseEncoder.create_from_config(config['critic']),
            discount_factor=config.get('discount_factor', 0.99),
            ratio_clip=config.get('ratio_clip', 0.2),
            actor_loss_weight=config.get('actor_loss_weight', 1.0),
            critic_loss_weight=config.get('critic_loss_weight', 1.0)
        )

    def forward(self, state: State) -> Tuple[State, Policy, Value, Value]:
        next_state, policy = self._actor.forward(state)
        curr_v_func = self._critic.forward(state)  # (batch_size)
        q_estimate = self._critic.forward(next_state)  # (batch_size)
        # TODO [Vladimir Baikalov]: make same inerface for every agent's forward function
        return next_state, policy, curr_v_func, q_estimate

    def loss(
            self,
            state: State,
            policy_old: Policy,
            start_v_old: Value,
            # Если есть трейсы разной длины видимо надо с маской что-то делать
            rewards: List[Reward],
            end_v_old: Value,
    ) -> Tensor:
        _, policy = self._actor.forward(state)
        policy_tensor = policy.to_tensor()
        policy_old_tensor = policy_old.to_tensor()
        prob_ratio = policy_tensor / policy_old_tensor
        start_v_old_tensor = start_v_old.to_tensor()

        advantage = self._compute_advantage_score(start_v_old, rewards, end_v_old)

        weighted_prob = advantage * prob_ratio
        weighted_clipped_prob = torch.clamp(prob_ratio, 1 - self.ratio_clip, 1 + self.ratio_clip) * advantage

        actor_loss = -torch.min(weighted_prob, weighted_clipped_prob)
        critic_loss = (advantage + start_v_old_tensor - self._critic.forward(state).to_tensor()) ** 2
        total_loss = self._actor_loss_weight * actor_loss + self._critic_loss_weight * critic_loss

        return total_loss

    def _compute_advantage_score(
            self,
            start_v_old: Value,
            rewards: List[Reward],
            end_v_old: Value
    ):
        end_v_old_tensor = end_v_old.to_tensor()
        advantage = start_v_old.to_tensor()
        for reward in rewards:
            advantage = self._discount_factor * advantage + reward.to_tensor()
        advantage -= end_v_old_tensor
        return advantage
