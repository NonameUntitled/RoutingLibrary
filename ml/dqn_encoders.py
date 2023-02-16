from abc import abstractmethod
from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import Categorical

from ml.encoders import TorchEncoder, TowerEncoder
from ml.utils import TensorWithMask


class BaseQNetwork(TorchEncoder, config_name='base_actor'):
    @abstractmethod
    def forward(
            self,
            current_node_idx: Tensor,
            neighbor_node_ids: TensorWithMask,
            destination_node_idx: Tensor,
            research_prob: float
    ) -> Tuple[Tensor, Tensor]:
        pass


def _with_random_research(
        argmax_next_neighbour: Tensor,
        neighbour: TensorWithMask,
        prob: float
):
    mask = torch.bernoulli(torch.full(argmax_next_neighbour.shape, prob)).int()
    reverse_mask = torch.ones(argmax_next_neighbour.shape).int() - mask

    uniform_weights = neighbour.mask
    uniform_distr = Categorical(probs=uniform_weights / uniform_weights.sum(dim=1))
    random_next_neighbour = neighbour.padded_values[torch.unsqueeze(uniform_distr.sample(), dim=1)]

    return argmax_next_neighbour * mask + random_next_neighbour * reverse_mask


class TowerQNetwork(BaseQNetwork, config_name='tower_actor'):
    def __init__(self, embedder: TorchEncoder, ff_net: TorchEncoder):
        super().__init__()
        self._ff_net = ff_net
        self._embedder = embedder

    @classmethod
    def create_from_config(cls, config):
        return cls(
            ff_net=TowerEncoder.create_from_config(config['ff_net']),
            embedder=TorchEncoder.create_from_config(config['embedder']),
        )

    def forward(
            self,
            current_node_idx: Tensor,
            neighbor_node_ids: TensorWithMask,
            destination_node_idx: Tensor,
            research_prob: float
    ) -> Tuple[Tensor, Tensor]:
        # 0) Create embeddings from indices
        # Shape: [batch_size, embedding_dim]
        current_node_embedding = self._embedder(current_node_idx)
        # Shape: [batch_size, embedding_dim]
        destination_node_embedding = self._embedder(destination_node_idx)

        # Shape: [all_batch_neighbors, embedding_dim]
        all_neighbors_embeddings = self._embedder(neighbor_node_ids.flatten_values)

        neighbor_node_embeddings = TensorWithMask(
            values=all_neighbors_embeddings,
            lengths=neighbor_node_ids.lengths
        )

        # Shape: [batch_size, max_neighbors_num, embedding_dim]
        padded_neighbors_node_embeddings = neighbor_node_embeddings.padded_values

        # 1) Shift neighbors and destinations
        # TODO[Vladimir Baikalov]: Make shifting optional
        # TODO[Vladimir Baikalov]: Check that it doesn't lead to gradient issues
        # Shape: [batch_size, embedding_dim]
        shifted_destination_node_embedding = destination_node_embedding - current_node_embedding
        # Shape: [batch_size, max_neighbors_num, embedding_dim]
        shifted_padded_neighbors_node_embeddings = \
            padded_neighbors_node_embeddings - torch.unsqueeze(current_node_embedding, dim=1)

        # 2) Compute q func
        # Shape: [batch_size, embedding_dim]
        neighbors_q = self._ff_net.forward(shifted_destination_node_embedding)

        # TODO[Vladimir Baikalov]: Probably it's a good idea to divide logits to make the distribution more smooth
        # TODO[Vladimir Baikalov]: Put constant in the variable
        neighbors_q[~neighbor_node_embeddings.mask] = -torch.inf

        next_neighbour_ids = neighbor_node_ids.flatten_values[torch.argmax(neighbors_q, dim=1)]
        next_neighbour = _with_random_research(
            next_neighbour_ids,
            neighbor_node_ids,
            research_prob
        )
        return next_neighbour, neighbors_q  # TODO move next search to agent?
