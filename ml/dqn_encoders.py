from abc import abstractmethod
from typing import Tuple

import torch
from torch import Tensor

from ml.encoders import TorchEncoder, TowerEncoder
from ml.functions import concat, argmax, softmax, sample
from ml.typing import TensorWithMask


class BaseQNetwork(TorchEncoder, config_name='base_actor'):
    @abstractmethod
    def forward(
            self,
            node_idx: Tensor,
            neighbour: TensorWithMask,
            destination: Tensor,
            research_prob: float,
            storage
    ) -> Tuple[Tensor, Tensor]:
        pass


def _with_random_research(
        argmax_next_neighbour: Tensor,
        neighbour: TensorWithMask,
        prob: float
):
    mask = torch.bernoulli(torch.full(argmax_next_neighbour.shape, prob)).int()
    reverse_mask = torch.ones(argmax_next_neighbour.shape).int() - mask
    masked_uniform_probs = softmax(TensorWithMask(torch.ones(neighbour.tensor.shape), neighbour.mask), dim=1)
    random_next_neighbour = sample(neighbour.tensor, masked_uniform_probs)
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
            node_idx: Tensor,
            neighbour: TensorWithMask,
            destination: Tensor,
            research_prob: float,
            storage
    ) -> Tuple[Tensor, Tensor]:
        current_embs = self._embedder.forward(TensorWithMask(node_idx), storage)
        neighbour_embs = self._embedder.forward(neighbour, storage)
        destination_embs = self._embedder.forward(TensorWithMask(destination), storage)

        shifted_neighbours = TensorWithMask(neighbour_embs.tensor - current_embs.tensor, neighbour_embs.mask)
        shifted_destination = TensorWithMask(destination_embs.tensor - current_embs.tensor, destination_embs.mask)
        neighbour_q = self._ff_net.forward(concat([shifted_neighbours, shifted_destination], dim=1))

        next_neighbour = neighbour.tensor[argmax(neighbour_q, dim=1)]
        next_neighbour = _with_random_research(
            next_neighbour,
            neighbour,
            research_prob
        )
        return next_neighbour, neighbour_q  # TODO move next search to agent?
