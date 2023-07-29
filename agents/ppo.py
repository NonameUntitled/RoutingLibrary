import copy
from typing import Dict, Any, Callable, Optional

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

import utils
from agents import TorchAgent
from ml import BaseOptimizer
from ml.encoders import BaseEncoder
from ml.ppo_encoders import BaseActor, BaseCritic
from ml.utils import TensorWithMask
from topology.utils import only_reachable_from
from utils.bag_trajectory import BaseBagTrajectoryMemory


EXP_CLIP = 300


def _get_logprob(neighbors, neighbors_logits):
    inf_tensor = torch.zeros(neighbors_logits.shape)
    inf_tensor[~neighbors.mask] = -torch.inf
    neighbors_logits = neighbors_logits + inf_tensor
    neighbors_logprobs = torch.nn.functional.log_softmax(neighbors_logits, dim=1)
    return neighbors_logprobs + inf_tensor


class PPOAgent(TorchAgent, config_name='ppo'):
    def __init__(
            self,
            current_node_idx_prefix: str,
            destination_node_idx_prefix: str,
            neighbors_node_ids_prefix: str,
            output_prefix: str,
            bag_ids_prefix: str,
            actor: BaseActor,
            critic: BaseCritic,
            freeze_weights: bool = False,
            actor_loss_weight: float = 1.0,
            critic_loss_weight: float = 1.0,
            entropy_loss_weight: float = 0.01,
            discount_factor: float = 0.99,
            ratio_clip: float = 0.2,
            bag_trajectory_memory: BaseBagTrajectoryMemory = None,  # Used only in training regime
            trajectory_length: int = 5,  # Used only in training regime
            trajectory_sample_size: int = 30,  # Used only in training regime
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

        self._freeze_weights = freeze_weights

        self._actor_loss_weight = actor_loss_weight
        self._critic_loss_weight = critic_loss_weight
        self._entropy_loss_weight = entropy_loss_weight
        self._discount_factor = discount_factor
        self._ratio_clip = ratio_clip

        self._bag_trajectory_memory = bag_trajectory_memory
        self._trajectory_length = trajectory_length
        self._trajectory_sample_size = trajectory_sample_size
        self._optimizer = optimizer_factory(self) if optimizer_factory is not None else None
        self._optimizer_factory = optimizer_factory

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
            freeze_weights=config.get('freeze_weights', False),
            actor_loss_weight=config.get('actor_loss_weight', 1.0),
            critic_loss_weight=config.get('critic_loss_weight', 1.0),
            entropy_loss_weight=config.get('entropy_loss_weight', 0.01),
            discount_factor=config.get('discount_factor', 0.99),
            ratio_clip=config.get('ratio_clip', 0.2),
            bag_trajectory_memory=BaseBagTrajectoryMemory.create_from_config(config['path_memory'])
            if 'path_memory' in config else None,
            trajectory_length=config.get('trajectory_length', 10),
            trajectory_sample_size=config.get('trajectory_sample_size', 30),
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
        next_neighbor_ids, neighbors_logits, neighbors_probs = self._actor(
            current_node_idx=current_node_idx,
            neighbor_node_ids=neighbor_node_ids,
            destination_node_idx=destination_node_idx
        )

        # Shape: [batch_size]
        current_state_value_function = self._critic(
            current_node_idx=current_node_idx,
            destination_node_idx=destination_node_idx
        )

        next_state_value_function = self._critic(
            current_node_idx=next_neighbor_ids,
            destination_node_idx=destination_node_idx
        )

        if self._bag_trajectory_memory is not None:
            # TODO[Vladimir Baikalov]: Think about how to generalize
            # Shape: [batch_size] if exists, None otherwise
            bag_ids = inputs.get(self._bag_ids_prefix, None)

            self._bag_trajectory_memory.add_to_trajectory(
                bag_ids=bag_ids,
                node_idxs=torch.full([batch_size], self.node_id),
                extra_infos=zip(
                    torch.unsqueeze(current_node_idx, dim=1),
                    neighbor_node_ids,
                    torch.unsqueeze(next_neighbor_ids.detach(), dim=1),
                    torch.unsqueeze(neighbors_logits.detach(), dim=1),
                    torch.unsqueeze(destination_node_idx, dim=1),
                    torch.unsqueeze(next_state_value_function.detach(), dim=1),
                )
            )

        inputs[self._output_prefix] = next_neighbor_ids
        # TODO[Zhogov Alexandr] fix it
        inputs.update({
            'predicted_next_node_idx': next_neighbor_ids,
            'predicted_next_node_logits': neighbors_logits,  # [neighbor_node_ids.mask].flatten(),
            'predicted_current_state_v_value': current_state_value_function
        })
        return inputs

    def learn(self) -> Optional[Tensor]:
        if self._freeze_weights:
            return None
        loss = 0
        learn_trajectories = self._bag_trajectory_memory.sample_trajectories_for_node_idx(
            node_idx=self.node_id,
            count=self._trajectory_sample_size,
            length=self._trajectory_length
        )
        if not learn_trajectories:
            return None
        for trajectory in learn_trajectories:
            loss += self._trajectory_loss(trajectory)
        loss /= len(learn_trajectories)
        self._optimizer.step(loss)
        return loss.detach().item()

    def _trajectory_loss(self, trajectory):
        rewards = [reward for _, reward in trajectory]
        parts = [part for part, _ in trajectory]

        node_idx, neighbors, next_neighbor, neighbor_logits_old, destination, _ = parts[0].extra_info
        _, _, _, _, _, end_v_old = parts[-1].extra_info
        if parts[-1].terminal:
            end_v_old = 0
        else:
            rewards = rewards[:-1]
        _, neighbor_logits, _ = self._actor(
            current_node_idx=node_idx,
            neighbor_node_ids=neighbors,
            destination_node_idx=destination
        )

        logprob_old = _get_logprob(neighbors, neighbor_logits_old)
        logprob = _get_logprob(neighbors, neighbor_logits)
        entropy = Categorical(logits=neighbor_logits).entropy()
        next_logprob_old = logprob_old[neighbors.padded_values == torch.unsqueeze(next_neighbor, dim=1)]
        next_logprob = logprob[neighbors.padded_values == torch.unsqueeze(next_neighbor, dim=1)]

        v = self._critic(
            current_node_idx=node_idx,
            destination_node_idx=destination
        )

        if neighbor_logits.shape[1] == 1:
            entropy = None

        return self._loss(next_logprob, next_logprob_old, v, end_v_old, rewards, entropy)

    def _loss(
            self,
            next_logprob,
            next_logprob_old,
            v,
            end_v_old,
            reward,
            entropy
    ) -> Tensor:
        prob_ratio = torch.exp(torch.clamp(next_logprob - next_logprob_old, -EXP_CLIP, EXP_CLIP))

        advantage = self._compute_advantage_score(v.detach(), reward, end_v_old)
        weighted_prob = prob_ratio * advantage
        weighted_clipped_prob = torch.clamp(prob_ratio, 1 - self._ratio_clip, 1 + self._ratio_clip) * advantage

        actor_loss = -torch.min(weighted_prob, weighted_clipped_prob)
        critic_loss = (self._compute_advantage_score(v, reward, end_v_old)) ** 2

        total_loss = self._actor_loss_weight * actor_loss
        total_loss += self._critic_loss_weight * critic_loss
        if entropy is not None:
            total_loss -= self._entropy_loss_weight * entropy

        return total_loss

    def _compute_advantage_score(
            self,
            v,
            rewards,
            end_v
    ):
        advantage = end_v
        for reward in rewards[::-1]:
            advantage = self._discount_factor * advantage + reward
        return advantage - v

    def debug(self, topology, step_num):
        nodes_list = sorted(topology.graph.nodes)
        nodes = {node: idx for idx, node in enumerate(nodes_list)}
        sinks = list(filter(lambda n: n.type == 'sink', topology.graph.nodes))
        diverter = nodes_list[self.node_id]
        current_node_idx = torch.LongTensor([nodes[diverter]])
        for sink in sinks:
            all_neighbors = list(topology.graph.successors(diverter))
            neighbors = only_reachable_from(
                graph=topology.graph,
                final_node=sink,
                start_nodes=all_neighbors
            )
            if not neighbors:
                continue
            neighbor_node_ids = TensorWithMask(torch.LongTensor([nodes[n] for n in neighbors]), torch.LongTensor([len(neighbors)]))
            destination_node_idx = torch.LongTensor([nodes[sink]])
            _, _, neighbors_probs = self._actor(
                current_node_idx=current_node_idx,
                neighbor_node_ids=neighbor_node_ids,
                destination_node_idx=destination_node_idx
            )
            current_state_value_function = self._critic(
                current_node_idx=current_node_idx,
                destination_node_idx=destination_node_idx
            )
            if utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER:
                utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.add_scalar(
                    '{}/{}'.format('train', f'v_func_{diverter}_{sink}'),
                    current_state_value_function.item(),
                    step_num
                )
                utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.flush()
                for idx, p in enumerate(neighbors_probs[0]):
                    utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.add_scalar(
                        '{}/{}'.format('train', f'prob_{diverter}_{neighbors[idx]}_{sink}'),
                        p.item(),
                        step_num
                      )
                    utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.flush()

    def _copy(self):
        agent_copy = copy.deepcopy(self)
        agent_copy._optimizer = agent_copy._optimizer_factory(agent_copy)
        return agent_copy
