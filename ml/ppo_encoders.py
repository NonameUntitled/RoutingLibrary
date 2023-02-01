from abc import abstractmethod
from typing import Tuple

from torch import Tensor

from ml.encoders import TorchEncoder, TowerEncoder
from ml.functions import concat, euclidean_distance, softmax, sample
from ml.typing import TensorWithMask


class BaseActor(TorchEncoder, config_name='base_actor'):
    @abstractmethod
    def forward(
            self,
            node: Tensor,
            neighbour: TensorWithMask,
            destination: Tensor,
            storage
    ) -> Tuple[Tensor, Tensor]:
        pass


class BaseCritic(TorchEncoder, config_name='base_critic'):
    @abstractmethod
    def forward(
            self,
            node: Tensor,
            destination: Tensor,
            storage
    ) -> Tensor:
        pass


class TowerActor(BaseActor, config_name='tower_actor'):
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
            node: Tensor,
            neighbour: TensorWithMask,
            destination: Tensor,
            storage
    ) -> Tuple[Tensor, Tensor]:
        current_embs = self._embedder.forward(TensorWithMask(node), storage)
        neighbour_embs = self._embedder.forward(neighbour, storage)
        destination_embs = self._embedder.forward(TensorWithMask(destination), storage)

        shifted_neighbours = TensorWithMask(neighbour_embs.tensor - current_embs.tensor, neighbour_embs.mask)
        shifted_destination = TensorWithMask(destination_embs.tensor - current_embs.tensor, destination_embs.mask)
        neighbour_embs = self._ff_net.forward(concat([shifted_neighbours, shifted_destination], dim=1)).tensor

        neighbour_logits = euclidean_distance(neighbour_embs, destination_embs, dim=1)
        neighbour_probs = softmax(TensorWithMask(1 / (neighbour_logits.tensor + 1e-18), neighbour_logits.mask), dim=1)
        next_neighbour = sample(neighbour.tensor, neighbour_probs)
        return next_neighbour, neighbour_probs


class TowerCritic(BaseCritic, config_name='tower_critic'):
    def __init__(self, embedder: TorchEncoder, ff_net: TorchEncoder):
        super().__init__()
        self._ff_net = ff_net
        self._embedder = embedder

    @classmethod
    def create_from_config(cls, config):
        return cls(
            ff_net=TowerEncoder.create_from_config(config['ff_net']),
            embedder=TorchEncoder.create_from_config(config['embedder'])
        )

    def forward(
            self,
            node: Tensor,
            destination: Tensor,
            storage
    ) -> Tensor:
        current_embs = self._embedder.forward(TensorWithMask(node), storage)
        destination_embs = self._embedder.forward(TensorWithMask(destination), storage)
        shifted_destination = TensorWithMask(destination_embs.tensor - current_embs.tensor, destination_embs.mask)
        return self._ff_net.forward(shifted_destination)
