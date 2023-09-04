from abc import abstractmethod
from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import Categorical

from ml import BaseEmbedding
from ml.encoders import TorchEncoder, TowerEncoder
from ml.utils import TensorWithMask


class BaseReinforceNetwork(TorchEncoder):
    @abstractmethod
    def forward(
            self,
            current_node_idx: Tensor,
            neighbor_node_ids: TensorWithMask,
            destination_node_idx: Tensor
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class TowerReinforceNetwork(BaseReinforceNetwork, config_name='tower_reinforce_network'):
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
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # 0) Create embeddings from indices
        # Shape: [batch_size, 16]
        current_node_embedding = self._embedder(current_node_idx.cpu())
        # Shape: [batch_size, 16]
        destination_node_embedding = self._embedder(destination_node_idx)
        # Shape: [all_batch_neighbors, 16]
        all_neighbors_embeddings = self._embedder(neighbor_node_ids.flatten_values)

        neighbor_node_embeddings = TensorWithMask(
            values=all_neighbors_embeddings,
            lengths=neighbor_node_ids.lengths
        )
        # Shape: [?, 2, 16]
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

        # 2) Compute neighbors similarities
        # Shape: [batch_size, embedding_dim]
        predicted_neighbors_embeddings = self._ff_net.forward(current_state_embedding)

        # Shape: [batch_size, max_neighbors_num]
        # TODO[Vladimir Baikalov]: Try cosine similarity as well
        neighbors_similarity_score = torch.einsum(
            'bnd,bd->bn',
            next_state_embeddings,
            predicted_neighbors_embeddings
        )
        neighbors_similarity_score[~neighbor_node_embeddings.mask] = -torch.inf

        # 3) Get probs from similarity scores
        # Shape: [batch_size, max_neighbors_num]
        neighbors_probs = torch.nn.functional.softmax(neighbors_similarity_score, dim=1)
        neighbors_probs = neighbors_probs * neighbor_node_embeddings.mask

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

        return next_neighbor_ids, torch.log(
            torch.gather(neighbors_probs, dim=1, index=next_neighbor_idx)
        ), neighbors_similarity_score
