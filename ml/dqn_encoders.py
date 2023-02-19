from abc import abstractmethod
from typing import Tuple

import torch
from torch import Tensor

from ml import BaseEmbedding
from ml.encoders import TorchEncoder, TowerEncoder
from ml.utils import TensorWithMask


class BaseQNetwork(TorchEncoder, config_name='base_q_network'):
    @abstractmethod
    def forward(
            self,
            current_node_idx: Tensor,
            neighbor_node_ids: TensorWithMask,
            destination_node_idx: Tensor
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class TowerQNetwork(BaseQNetwork, config_name='tower_q_network'):
    def __init__(self, embedder: TorchEncoder, ff_net: TorchEncoder):
        super().__init__()
        self._ff_net = ff_net
        self._embedder = embedder

    @classmethod
    def create_from_config(cls, config):
        return cls(
            ff_net=TowerEncoder.create_from_config(config['ff_net']),
            embedder=BaseEmbedding.create_from_config(config['embedder'])
        )

    def forward(
            self,
            current_node_idx: Tensor,
            neighbor_node_ids: TensorWithMask,
            destination_node_idx: Tensor
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
        neighbors_q[~neighbor_node_embeddings.mask] = -1e18
        return neighbors_q
