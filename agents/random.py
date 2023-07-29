import copy
from typing import Dict

import torch

from agents import TorchAgent


class RandomAgent(TorchAgent, config_name='random'):

    def __init__(
            self,
            neighbors_prefix: str,
            output_prefix: str = None
    ):
        super().__init__()
        self._neighbors_prefix = neighbors_prefix
        self._output_prefix = output_prefix or neighbors_prefix

    def forward(self, inputs: Dict) -> Dict:
        neighbors = inputs[self._neighbors_prefix]  # (batch_size, neighbors_cnt)

        neighbors_length = neighbors.lengths  # (batch_size)

        low = torch.zeros_like(neighbors_length)  # (batch_size)
        high = neighbors_length  # (batch_size)
        next_neighbors = torch.round(torch.rand(neighbors_length.shape) * (neighbors_length - 1)).long()  # (batch_size)

        next_neighbors = torch.squeeze(
            torch.gather(
                input=neighbors.flatten_values,  # (batch_size, num_neighbors)
                dim=1,
                index=torch.unsqueeze(next_neighbors, dim=1)  # (batch_size, 1)
            )  # (batch_size, 1)
        )  # (batch_size)

        inputs[self._output_prefix] = next_neighbors

        return inputs

    @classmethod
    def create_from_config(cls, config):
        return cls(
            neighbors_prefix=config['neighbors_prefix'],
            output_prefix=config['output_prefix']
        )

    def learn(self):
        pass

    def _copy(self):
        return copy.deepcopy(self)
