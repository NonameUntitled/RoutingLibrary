from abc import abstractmethod
from typing import Tuple

import torch
from torch import Tensor
from torch.distributions.categorical import Categorical

from ml.embeddings import BaseEmbedding
from ml.encoders import TorchEncoder, TowerEncoder
from ml.utils import TensorWithMask, BIG_NEG


class BaseActor(TorchEncoder):

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

    def __init__(
            self,
            embedder: TorchEncoder,
            ff_net: TorchEncoder,
            use_embedding_shift: bool = True,
            logits_scale: float = 4.0
    ):
        super().__init__()
        self._embedder = embedder
        self._ff_net = ff_net
        self._use_embedding_shift = use_embedding_shift
        self._logits_scale = logits_scale

    @classmethod
    def create_from_config(cls, config):
        return cls(
            embedder=BaseEmbedding.create_from_config(config['embedder']),
            ff_net=TowerEncoder.create_from_config(config['ff_net']),
            use_embedding_shift=config.get('use_embedding_shift', True),
            logits_scale=config.get('logits_scale', 4.0)
        )

    def forward(
            self,
            current_node_idx: Tensor,
            neighbor_node_ids: TensorWithMask,
            destination_node_idx: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
        # TODO[Vladimir Baikalov]: Check that it doesn't lead to gradient issues
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

        # 2) Compute representation of next ideal state
        # Shape: [batch_size, embedding_dim]
        ideal_transition_embedding = self._ff_net.forward(current_state_embedding)

        # 3) Compute logits for existing next states (here I use dot product for scores receiving)
        # Shape: [batch_size, max_neighbors_num]

        neighbors_logits = torch.nn.functional.cosine_similarity(
            next_state_embeddings,
            torch.zeros(next_state_embeddings.shape) + ideal_transition_embedding[:, None, :], dim=2
        ) * self._logits_scale

        # neighbors_logits = torch.einsum(
        #     'bnd,bd->bn',
        #     next_state_embeddings,
        #     ideal_transition_embedding
        # ) / 10.0

        # TODO[Vladimir Baikalov]: Probably it's a good idea to divide logits to make the distribution smoother
        inf_tensor = torch.zeros(neighbors_logits.shape)
        inf_tensor[~neighbor_node_embeddings.mask] = BIG_NEG
        neighbors_logits = neighbors_logits + inf_tensor

        # 4) Get probs from logits
        # Shape: [batch_size, max_neighbors_num]
        neighbors_probs = torch.nn.functional.softmax(neighbors_logits, dim=1)
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

        return next_neighbor_ids, neighbors_logits, neighbors_probs


class TowerCritic(BaseCritic, config_name='tower_critic'):

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
            destination_node_idx: Tensor
    ) -> Tensor:
        # 0) Create embeddings from indices
        # Shape: [batch_size, embedding_dim]
        current_node_embedding = self._embedder(current_node_idx)
        # Shape: [batch_size, embedding_dim]
        destination_node_embedding = self._embedder(destination_node_idx)

        # 1) Create representation for current state and next states
        if self._use_embedding_shift:
            # Shape: [batch_size, embedding_dim]
            current_state_embedding = destination_node_embedding - current_node_embedding
        else:
            # Shape: [batch_size, 2 * embedding_dim]
            current_state_embedding = torch.cat(
                [destination_node_embedding, current_node_embedding],
                dim=-1
            )

        # 2) Compute value function for current state
        # Shape: [batch_size]
        current_state_value_function = torch.squeeze(
            self._ff_net.forward(current_state_embedding),
            dim=1
        )

        return current_state_value_function
