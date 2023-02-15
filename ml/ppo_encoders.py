from abc import abstractmethod
from typing import Tuple

import torch
from torch import Tensor
from torch.distributions.categorical import Categorical

from ml.embeddings import BaseEmbedding
from ml.encoders import TorchEncoder, TowerEncoder
from ml.utils import TensorWithMask


class BaseActor(TorchEncoder, config_name='base_actor'):

    @abstractmethod
    def forward(
            self,
            current_node_idx: Tensor,
            neighbor_node_ids: TensorWithMask,
            destination_node_idx: Tensor
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class BaseCritic(TorchEncoder, config_name='base_critic'):

    @abstractmethod
    def forward(
            self,
            current_node_idx: Tensor,
            destination_node_idx: Tensor
    ) -> Tensor:
        raise NotImplementedError


class TowerActor(BaseActor, config_name='tower_actor'):

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

        # 2) Compute embedding of ideal transition
        # Shape: [batch_size, embedding_dim]
        ideal_transition_embedding = self._ff_net.forward(shifted_destination_node_embedding)

        # 3) Compute logits for existing neighbors
        # Shape: [batch_size, max_neighbors_num]
        neighbors_logits = torch.einsum(
            'bnd,bd->bn',
            shifted_padded_neighbors_node_embeddings,
            ideal_transition_embedding
        )
        # TODO[Vladimir Baikalov]: Probably it's a good idea to divide logits to make the distribution more smooth
        # TODO[Vladimir Baikalov]: Put constant in the variable
        neighbors_logits[~neighbor_node_embeddings.mask] = -1e18

        # 4) Get probs from logits
        # Shape: [batch_size, max_neighbors_num]
        neighbors_probs = torch.nn.functional.softmax(neighbors_logits, dim=1)
        neighbors_probs[~neighbor_node_embeddings.mask] = 0

        # TODO[Vladimir Baikalov]: Check these two lines below
        neighbors_probs /= neighbors_probs.max()
        neighbors_probs[~neighbor_node_embeddings.mask] = 0

        # 4) Sample next neighbor idx
        categorical_distribution = Categorical(probs=neighbors_probs)
        # Shape: [batch_size, 1]
        next_neighbors_ids = torch.unsqueeze(categorical_distribution.sample(), dim=1)

        # Shape: [batch_size]
        next_neighbors_ids = torch.squeeze(torch.gather(
            neighbor_node_ids.padded_values,
            dim=1,
            index=next_neighbors_ids
        ))

        return next_neighbors_ids, neighbors_logits


class TowerCritic(BaseCritic, config_name='tower_critic'):

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
            destination_node_idx: Tensor
    ) -> Tensor:
        # 0) Create embeddings from indices
        # Shape: [batch_size, embedding_dim]
        current_node_embedding = self._embedder(current_node_idx)
        # Shape: [batch_size, embedding_dim]
        destination_node_embedding = self._embedder(destination_node_idx)

        # 1) Shift destination
        # Shape: [batch_size, embedding_dim]
        shifted_destination_node_embedding = destination_node_embedding - current_node_embedding

        # 2) Compute value function for current state
        # Shape: [batch_size]
        current_state_value_function = torch.squeeze(
            self._ff_net.forward(shifted_destination_node_embedding)
        )

        return current_state_value_function
