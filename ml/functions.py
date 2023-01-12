from typing import List

import torch
from torch import Tensor

from ml.typing import TensorWithMask


def concat(tensors: List[TensorWithMask], dim):
    embeddings = []
    masks = []

    for tensor_with_mask in tensors:
        embeddings.append(tensor_with_mask.tensor)
        masks.append(tensor_with_mask.mask)

    return TensorWithMask(
        torch.cat(embeddings, dim=dim),
        torch.cat(masks, dim=dim)
    )


def softmax(inputs: TensorWithMask, dim) -> Tensor:
    scores = inputs.tensor
    mask = inputs.mask

    scores[~mask] = -1e18  # TODO [Vladimir Baikalov]: use torch INF here
    probabilities = torch.softmax(scores, dim=dim)
    return probabilities


def euclidean_distance(left_tensor: TensorWithMask, right_tensor: Tensor, dim) -> TensorWithMask:
    if len(left_tensor.tensor.shape) == len(right_tensor.shape) + 1:
        right_tensor = right_tensor[None]
    distances = torch.sqrt(torch.sum((left_tensor.tensor - right_tensor) ** 2, dim=dim+1))
    distances[left_tensor.mask] = 1e18
    return TensorWithMask(distances, left_tensor.mask)


def sample(values: Tensor, probs: Tensor) -> Tensor:
    distr = torch.distributions.Categorical(probs)
    # TODO what dims?
    return values[distr.sample()]
