from typing import List, Tuple
import torch

from agents import Policy, State, Value, Reward, TorchAgent
from agents.actor import BaseActor
from agents.critic import BaseCritic


class PPOAgent(TorchAgent, config_name='ppo'):
    def __init__(
            self, actor: BaseActor,
            critic: BaseCritic,
            discount_factor: float,
            ratio_clip: float
    ):
        super().__init__()
        self._actor = actor
        self._critic = critic
        self._discount_factor = discount_factor
        self._ratio_clip = ratio_clip

    @classmethod
    def create_from_config(cls, config):
        return cls(
            actor=BaseActor.create_from_config(config['actor']),
            critic=BaseCritic.create_from_config(config['critic']),
            discount_factor=config.get('discount_factor', 0.99),
            ratio_clip=config.get('ratio_clip', 0.2)
        )

    def forward(self, state: State) -> Tuple[State, Policy, Value, Value]:
        next_state, policy = self._actor.forward(state)
        curr_v_func = self._critic.forward(state)
        q_estimate = self._critic.forward(next_state)
        return next_state, policy, curr_v_func, q_estimate

    def loss(
            self,
            state: State,
            policy_old: Policy,
            start_v_old: Value,
            # Если есть трейсы разной длины видимо надо с маской что-то делать
            rewards: List[Reward],
            end_v_old: Value,
    ) -> None:
        _, policy = self._actor.forward(state)
        policy_tensor = policy.to_tensor()
        policy_old_tensor = policy_old.to_tensor()
        prob_ratio = policy_tensor / policy_old_tensor
        start_v_old_tensor = start_v_old.to_tensor()
        end_v_old_tensor = end_v_old.to_tensor()
        advantage = start_v_old_tensor
        for reward in rewards:
            advantage = self._discount_factor * advantage + reward.to_tensor()
        advantage -= end_v_old_tensor
        weighted_prob = advantage * prob_ratio
        weighted_clipped_prob = torch.clamp(prob_ratio, 1 - self.ratio_clip, 1 + self.ratio_clip) * advantage
        actor_loss = -torch.min(weighted_prob, weighted_clipped_prob)
        critic_loss = (advantage + start_v_old_tensor - self._critic.forward(state).to_tensor()) ** 2
        return actor_loss, critic_loss
