from abc import abstractmethod
from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import Categorical

from ml import BaseEmbedding
from ml.encoders import TorchEncoder, TowerEncoder
from ml.utils import TensorWithMask, BIG_NEG


class BaseQNetwork(TorchEncoder):
    @abstractmethod
    def forward(
            self,
            current_node_idx: Tensor,
            neighbor_node_ids: TensorWithMask,
            destination_node_idx: Tensor
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class TowerQNetwork(BaseQNetwork, config_name='tower_q_network'):
    def __init__(
            self,
            embedder: TorchEncoder,
            ff_net: TorchEncoder,
            use_embedding_shift: bool = True
    ):
        super().__init__()
        self._ff_net = ff_net
        self._embedder = embedder
        self._use_embedding_shift = use_embedding_shift

    @classmethod
    def create_from_config(cls, config):
        return cls(
            ff_net=TowerEncoder.create_from_config(config['ff_net']),
            embedder=BaseEmbedding.create_from_config(config['embedder']),
            use_embedding_shift=config.get('use_embedding_shift', True)
        )

    def forward(
            self,
            current_node_idx: Tensor,
            neighbor_node_ids: TensorWithMask,
            destination_node_idx: Tensor
    ) -> (Tensor, Tensor):
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

        # 1) Create representation for current state and next states
        if self._use_embedding_shift:
            # Shape: [batch_size, embedding_dim]
            current_state_embedding = destination_node_embedding - current_node_embedding
            # Shape: [batch_size, max_neighbors_num, embedding_dim]
            next_state_embeddings = \
                padded_neighbors_node_embeddings - torch.unsqueeze(current_node_embedding, dim=1)
        else:
            # Shape: [batch_size, 2 * embedding_dim]
            current_state_embedding = torch.cat(
                [destination_node_embedding, current_node_embedding],
                dim=-1
            )
            # Shape: [batch_size, max_neighbors_num, embedding_dim]
            next_state_embeddings = padded_neighbors_node_embeddings
        # Shape: [batch_size, max_neighbors_num, 2 (1 for shifted) embedding_dim]
        current_state_embedding_expanded = \
            torch.repeat_interleave(
                torch.unsqueeze(current_state_embedding, dim=-2),
                next_state_embeddings.shape[-2],
                dim=-2
            )
        # Shape: [batch_size, max_neighbors_num, 3 (2 for shifted) * embedding_dim]
        all_state_embeddings = torch.cat(
            [current_state_embedding_expanded, next_state_embeddings],
            dim=-1
        )

        # 2) Compute q func
        # Shape: [batch_size, max_neighbors_num]
        neighbors_q = torch.squeeze(self._ff_net.forward(all_state_embeddings), dim=-1)
        # TODO[Vladimir Baikalov]: Probably it's a good idea to divide logits to make the distribution smoother
        inf_tensor = torch.zeros(neighbors_q.shape)
        inf_tensor[~neighbor_node_embeddings.mask] = -torch.inf
        neighbors_q = neighbors_q + inf_tensor

        # 3) Get probs from q-logits
        # Shape: [batch_size, max_neighbors_num]
        neighbors_probs = torch.nn.functional.softmax(neighbors_q, dim=1)
        neighbors_probs = neighbors_probs * neighbor_node_embeddings.mask  # Make sure we won't sample from padding

        # 4) Sample next neighbor idx
        categorical_distribution = Categorical(probs=neighbors_probs)
        # Shape: [batch_size, 1]
        next_neighbor_idx = torch.unsqueeze(categorical_distribution.sample(), dim=1)

        # Shape: [batch_size]
        next_neighbor_ids = torch.squeeze(torch.gather(
            neighbor_node_ids.padded_values,
            dim=1,
            index=next_neighbor_idx
        ), dim=1)

        return next_neighbor_ids, neighbors_q
