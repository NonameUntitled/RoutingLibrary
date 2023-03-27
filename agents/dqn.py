from copy import deepcopy
from typing import Dict, Any, Callable, Optional

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from agents import TorchAgent
from ml import BaseOptimizer
from ml.dqn_encoders import BaseQNetwork
from ml.encoders import BaseEncoder
from ml.utils import TensorWithMask
from utils.bag_trajectory import BaseBagTrajectoryMemory


def _with_random_research(
        argmax_next_neighbor: Tensor,
        neighbor: TensorWithMask,
        prob: float
):
    probs = torch.stack([
        torch.full(argmax_next_neighbor.shape, prob),
        torch.full(argmax_next_neighbor.shape, 1 - prob)
    ], dim=1)
    choice_idx = Categorical(probs=probs).sample()

    neighbor_tensor = neighbor.padded_values
    random_probs = torch.ones(neighbor_tensor.shape)
    random_probs[~neighbor.mask] = -torch.inf
    random_probs = torch.softmax(random_probs, dim=1)
    random_neighbours = torch.squeeze(torch.gather(
        neighbor_tensor,
        dim=1,
        index=torch.unsqueeze(Categorical(probs=random_probs).sample(), dim=1)
    ), dim=1)

    choices = torch.stack([
        argmax_next_neighbor,
        random_neighbours
    ], dim=1)
    return torch.squeeze(torch.gather(choices, dim=1, index=torch.unsqueeze(choice_idx, dim=1)), dim=1)


class DQNAgent(TorchAgent, config_name='dqn'):
    def __init__(
            self,
            current_node_idx_prefix: str,
            destination_node_idx_prefix: str,
            neighbors_node_ids_prefix: str,
            output_prefix: str,
            bag_ids_prefix: str,
            q_network: BaseQNetwork,
            discount_factor: float = 0.99,
            research_prob: float = 0.1,
            bag_trajectory_memory: BaseBagTrajectoryMemory = None,
            trajectory_sample_size: int = 30,
            optimizer_factory: Callable[[nn.Module], BaseOptimizer] = None
    ):
        assert 0 < discount_factor < 1, 'Incorrect `discount_factor` choice'
        assert 0 < research_prob < 1, 'Incorrect `discount_factor` choice'
        super().__init__()

        self._node_id = -1

        self._current_node_idx_prefix = current_node_idx_prefix
        self._destination_node_idx_prefix = destination_node_idx_prefix
        self._neighbors_node_ids_prefix = neighbors_node_ids_prefix
        self._output_prefix = output_prefix
        self._bag_ids_prefix = bag_ids_prefix

        self._q_network = q_network
        self._discount_factor = discount_factor
        self._research_prob = research_prob
        self._bag_trajectory_memory = bag_trajectory_memory
        self._trajectory_sample_size = trajectory_sample_size

        self._optimizer = optimizer_factory(self) if optimizer_factory is not None else None

        self._weight_dump = 3,
        self._weight_dump_counter = 0
        self._old_q_network = deepcopy(self._q_network)

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
            research_prob=config.get('research_prob', 0.1),
            bag_trajectory_memory=BaseBagTrajectoryMemory.create_from_config(config['path_memory'])
            if 'path_memory' in config else None,
            trajectory_sample_size=config.get('trajectory_sample_size', 30),
            optimizer_factory=lambda m: BaseOptimizer.create_from_config(config['optimizer'], model=m)
            if 'optimizer' in config else None
        )

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Shape: [batch_size]
        current_node_idx = inputs[self._current_node_idx_prefix]
        if self._node_id < 0:
            self._node_id = current_node_idx[0].item()
        batch_size = len(current_node_idx)
        # Shape: [batch_size]
        destination_node_idx = inputs[self._destination_node_idx_prefix]
        # TensorWithMask
        # Shape: [batch_size, max_neighbors_num]
        neighbor_node_ids = inputs[self._neighbors_node_ids_prefix]

        # Shape: [batch_size, max_neighbors_num]
        next_neighbor_q = self._q_network(
            current_node_idx=current_node_idx,
            neighbor_node_ids=neighbor_node_ids,
            destination_node_idx=destination_node_idx
        )

        # Shape: [batch_size]
        best_next_neighbor_id = torch.squeeze(torch.gather(
            neighbor_node_ids.padded_values,
            dim=1,
            index=torch.unsqueeze(torch.argmax(next_neighbor_q, dim=1), dim=1)
        ), dim=1)
        next_neighbor_ids = _with_random_research(
            best_next_neighbor_id,
            neighbor_node_ids,
            self._research_prob
        ).detach()

        if self._bag_trajectory_memory is not None:
            # TODO[Vladimir Baikalov]: Think about how to generalize
            bag_ids = inputs.get(self._bag_ids_prefix, None)

            self._bag_trajectory_memory.add_to_trajectory(
                bag_ids=bag_ids,
                node_idxs=torch.full([batch_size], self._node_id),
                extra_infos=zip(
                    torch.unsqueeze(current_node_idx, dim=1),
                    neighbor_node_ids,
                    torch.unsqueeze(destination_node_idx, dim=1),
                    torch.unsqueeze(next_neighbor_ids, dim=1)
                )
            )
        inputs[self._output_prefix] = next_neighbor_ids
        inputs.update({
            'predicted_next_node_idx': next_neighbor_ids,
            'predicted_next_node_q': next_neighbor_q,
        })
        return inputs

    def learn(self) -> Optional[Tensor]:
        loss = 0
        learn_trajectories = self._bag_trajectory_memory.sample_trajectories_for_node_idx(
            self._node_id,
            self._trajectory_sample_size,
            1
        )
        if not learn_trajectories:
            return None
        for trajectory in learn_trajectories:
            target = trajectory[0]['reward']
            current_node_idx, neighbor_node_ids, destination_node_idx, next_neighbor = trajectory[0]['extra_info']
            if len(trajectory) > 1:
                current_node_idx_, neighbor_node_ids_, destination_node_idx_, _ = trajectory[1]['extra_info']
                neighbor_q_ = self._old_q_network(
                    current_node_idx=current_node_idx_,
                    neighbor_node_ids=neighbor_node_ids_,
                    destination_node_idx=destination_node_idx_
                )
                target += self._discount_factor * torch.max(neighbor_q_, dim=1).values.detach()
            neighbor_q = self._q_network(
                current_node_idx=current_node_idx,
                neighbor_node_ids=neighbor_node_ids,
                destination_node_idx=destination_node_idx
            )
            loss += (target - neighbor_q[neighbor_node_ids.padded_values == next_neighbor]) ** 2
        loss /= self._trajectory_sample_size
        self._optimizer.step(loss)

        self._weight_dump_counter += 1
        if self._weight_dump_counter == self._weight_dump:
            self._weight_dump_counter = 0
            for curr_param, old_param in zip(self._q_network.parameters(), self._old_q_network.parameters()):
                old_param.data.copy_(curr_param.data)

        return loss.detach().item()
