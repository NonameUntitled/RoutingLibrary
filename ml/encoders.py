import torch
import torch.nn as nn

from typing import List

from ml import BaseEncoder, BaseDistance, TensorWithMask


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


class TorchDistance(BaseDistance, nn.Module):

    def __init__(self):
        super().__init__()


class SharedEmbeddingEncoder(TorchEncoder, config_name='shared_embedding_encoder'):
    """
    Returns embeddings for given nodes given as a nodes indices `prefix`
    from a shared embeddings storage `embeddings_prefix`.
    Storage should return embedding by a `__call__()` build-in method call (This line is valid only for torch storage)
    """

    def __init__(
            self,
            prefix: str,
            storage_prefix: str,
            output_prefix: str = None
    ):
        super().__init__()
        self._prefix = prefix
        self._storage_prefix = storage_prefix
        self._output_prefix = output_prefix or prefix

    def forward(self, inputs):
        nodes = inputs[self._prefix]  # (batch_size, nodes_cnt)
        nodes_mask = inputs[f'{self._prefix}.mask']  # (batch_size, nodes_cnt)
        storage = inputs[self._storage_prefix]  # EmbeddingStorage

        all_nodes = nodes[nodes_mask]  # (all_nodes_in_batch)

        batch_size = nodes.shape[0]
        nodes_cnt = nodes.shape[1]
        embedding_dim = storage.embedding_dim

        nodes_embeddings = torch.zeros(
            batch_size, nodes_cnt, embedding_dim,
            dtype=torch.float, device=nodes.device
        )  # (batch_size, nodes_cnt, emb_dim)
        nodes_embeddings[nodes_mask] = storage(all_nodes)  # (batch_size, nodes_cnt, emb_dim)

        inputs[self._output_prefix] = nodes_embeddings[nodes_mask]  # (batch_size, nodes_cnt, emb_dim)
        inputs[f'{self.__output_prefix}.mask'] = nodes_mask  # (batch_size, nodes_cnt)

        return inputs


class SubtractionEncoder(TorchEncoder, config_name='subtraction'):

    def __init__(
        self,
        left_operand_prefix: str,
        right_operand_prefix: str,
        output_prefix: str
    ):
        super().__init__()
        self._left_operand_prefix = left_operand_prefix
        self._right_operand_prefix = right_operand_prefix
        self._output_prefix = output_prefix

    def forward(self, inputs):
        left_operand = inputs[self._left_operand_prefix]  # (batch_size, ???)
        left_operand_mask = inputs[f'{self._left_operand_prefix}.mask']  # (batch_size, ???)

        right_operand = inputs[self._right_operand_prefix]  # (batch_size, ???)
        right_operand_mask = inputs[f'{self._right_operand_prefix}.mask']  # (batch_size, ???)

        result = torch.zeros(
            *left_operand.shape,
            dtype=left_operand.dtype,
            device=left_operand.device
        )  # (batch_size, ???)
        result[left_operand_mask] = torch.subtract(left_operand[left_operand_mask], right_operand[right_operand_mask])

        inputs[self._output_prefix] = result  # (batch_size, ???)
        inputs[f'{self._output_prefix}.mask'] = left_operand_mask  # (batch_size, ???)

        return result


class ConcatEncoder(TorchEncoder, config_name='concat'):

    def __init__(
            self,
            input_prefixes: List[str],
            dim: int,
            output_prefix: str,
            concat_masks: bool = True
    ):
        super().__init__()
        self._input_prefixes = input_prefixes
        self._dim = dim
        self._output_prefix = output_prefix
        self._concat_masks = concat_masks

    def forward(self, inputs):
        embeddings = []
        masks = []

        for prefix in self._prefixes:
            embed = inputs[prefix]
            mask = inputs[f'{prefix}.mask']

            embeddings.append(embed)
            masks.append(mask)

        inputs[self._output_prefix] = torch.cat(embeddings, dim=self._dim)
        if self._concat_masks:
            inputs[f'{self._output_prefix}.mask'] = torch.cat(masks, dim=self._dim)

        return inputs


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
            output_prefix: str = None,
            input_dim: int = None,
            output_dim: int = None,
            dropout: float = 0.,
            initializer_range: float = 0.02,
            eps: float = 1e-12
    ):
        super().__init__()
        self._hidden_sizes = hidden_dims
        self._output_prefix = output_prefix or self._prefix

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

    def forward(self, inputs: TensorWithMask):
        embeddings = inputs.tensor

        embeddings = self._input_projector(embeddings)
        embeddings = self._layers(embeddings)
        embeddings = self._output_projector(embeddings)

        return TensorWithMask(embeddings, embeddings[inputs.mask])


class SoftmaxEncoder(TorchEncoder, config_name='softmax'):

    def __init__(
            self,
            prefix: str,
            dim: int,
            output_prefix: str = None
    ):
        super().__init__()
        self._prefix = prefix
        self._dim = dim
        self._output_prefix = output_prefix or prefix

    def forward(self, inputs):
        scores = inputs[self._prefix]
        mask = inputs[f'{self._prefix}.mask']

        scores[~mask] = -1e18  # TODO [Vladimir Baikalov]: use torch INF here

        probabilities = torch.softmax(scores, dim=self._dim)

        inputs[self._output_prefix] = probabilities
        inputs[f'{self._output_prefix}.mask'] = mask

        return inputs


class EuclideanDistance(TorchDistance, config_name='euclidean_distance'):

    def __init__(
            self,
            left_operand_prefix: str,
            right_operand_prefix: str,
            dim: int,
            output_prefix: str
    ):
        super().__init__()
        self._left_operand_prefix = left_operand_prefix
        self._right_operand_prefix = right_operand_prefix
        self._dim = dim
        self._output_prefix = output_prefix

    def forward(self, inputs):
        left_operand = inputs[self._left_operand_prefix]
        left_operand_mask = inputs[f'{self._left_operand_prefix}.mask']

        right_operand = inputs[self.__right_operand_prefix]

        assert right_operand.shape[self._dim] == 1 or right_operand.shape[self._dim] == left_operand.shape[self._dim]
        distances = torch.sqrt(torch.sum((left_operand - right_operand) ** 2, dim=self._dim))

        inputs[self._output_prefix] = distances
        inputs[self._output_prefix][~left_operand_mask] = 1e18

        inputs[f'{self._output_prefix}.mask'] = left_operand_mask

        return inputs
