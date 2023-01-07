from abc import abstractmethod

import torch


class TensorWithMask:
    def __init__(self, tensor: torch.Tensor, mask: torch.Tensor = None):
        self.tensor = tensor
        if mask is None:
            mask = torch.ones(tensor.shape, dtype=torch.bool)
        self.mask = mask


class TensorLike:
    @abstractmethod
    def to_tensor(self) -> torch.Tensor:
        pass
