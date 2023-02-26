from typing import Dict

import numpy as np
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

        neighbors_length = neighbors.lengths

        low = torch.zeros_like(neighbors_length)
        high = neighbors_length
        next_neighbors = torch.randint(low=low.item(), high=high.item(), size=(1,))

        next_neighbor = torch.gather(neighbors.flatten_values, 1, torch.unsqueeze(next_neighbors, 0))

        inputs[self._output_prefix] = next_neighbor

        return inputs

    @classmethod
    def create_from_config(cls, config):
        return cls(
            neighbors_prefix=config['neighbors_prefix'],
            output_prefix=config['output_prefix']
        )

    def learn(self):
        pass

