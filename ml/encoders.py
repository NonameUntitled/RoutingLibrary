from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from ml.typing import TensorWithMask
from utils import MetaParent


class BaseEncoder(metaclass=MetaParent):
    pass


class TorchEncoder(BaseEncoder, nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _init_weights(layer, initializer_range=0.02):
        nn.init.trunc_normal_(
            layer.weight,
            std=initializer_range,
            a=-2 * initializer_range,
            b=2 * initializer_range
        )
        nn.init.zeros_(layer.bias)


class SharedEmbeddingEncoder(TorchEncoder, config_name='shared_embedding_encoder'):
    def forward(self, inputs: TensorWithMask, storage) -> TensorWithMask:
        nodes = inputs.tensor  # (batch_size, nodes_cnt)
        nodes_mask = inputs.mask  # (batch_size, nodes_cnt)s

        all_nodes = nodes[nodes_mask]  # (all_nodes_in_batch)

        batch_size = nodes.shape[0]
        nodes_cnt = nodes.shape[1]
        embedding_dim = storage.embedding_dim

        nodes_embeddings = torch.zeros(
            batch_size, nodes_cnt, embedding_dim,
            dtype=torch.float, device=nodes.device
        )  # (batch_size, nodes_cnt, emb_dim)
        nodes_embeddings[nodes_mask] = storage(all_nodes)  # (batch_size, nodes_cnt, emb_dim)
        return TensorWithMask(nodes_embeddings, nodes_mask)


class TowerBlock(TorchEncoder, nn.Module):

    def __init__(
            self,
            input_dim,
            output_dim,
            eps,
            dropout,
            initializer_range
    ):
        super().__init__()
        self._linear = nn.Linear(input_dim, output_dim)
        self._relu = nn.ReLU()
        self._layernorm = nn.LayerNorm(output_dim, eps=eps)
        self._dropout = nn.Dropout(p=dropout)
        TorchEncoder._init_weights(self._linear, initializer_range)

    def forward(self, x):
        return self._layernorm(self._dropout(self._relu(self._linear(x))) + x)


class TowerEncoder(TorchEncoder, config_name='tower'):

    def __init__(
            self,
            hidden_dims: List[int],
            input_dim: int = None,
            output_dim: int = None,
            dropout: float = 0.,
            initializer_range: float = 0.02,
            eps: float = 1e-12
    ):
        super().__init__()
        self._hidden_sizes = hidden_dims

        self._input_projector = nn.Identity()
        if input_dim is not None:
            self._input_projector = nn.Linear(input_dim, hidden_dims[0])
            self._init_weights(self._input_projector, initializer_range)

        self._layers = nn.Sequential(
            *[
                TowerBlock(
                    input_dim=hidden_dims[i],
                    output_dim=hidden_dims[i + 1],
                    eps=eps,
                    dropout=dropout,
                    initializer_range=initializer_range
                )
                for i in range(len(hidden_dims) - 1)
            ]
        )

        self._output_projector = nn.Identity()
        if output_dim is not None:
            self._output_projector = nn.Linear(hidden_dims[-1], output_dim)
            TorchEncoder._init_weights(self._output_projector, initializer_range)

    def forward(self, inputs: TensorWithMask) -> Tensor:
        embeddings = inputs.tensor

        embeddings = self._input_projector(embeddings)
        embeddings = self._layers(embeddings)
        embeddings = self._output_projector(embeddings)

        return embeddings
