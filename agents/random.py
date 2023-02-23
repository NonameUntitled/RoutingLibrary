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

        neighbors_mask = neighbors.mask  # (batch_size, neighbors_cnt)
        neighbors_mask_sum = torch.sum(neighbors_mask, dim=-1)  # (batch_size)

        low = np.zeros_like(neighbors_mask_sum.cpu().detach().numpy())
        high = neighbors_mask_sum.cpu().detach().numpy()

        next_neighbors = torch.from_numpy(np.random.randint(
            low=low,
            high=high
        ))

        next_neighbors = next_neighbors.to(next_neighbors.device)  # (batch_size)

        next_neighbor = neighbors.flatten_values.numpy()[0][next_neighbors.item()]

        return next_neighbor

    @classmethod
    def create_from_config(cls, config):
        return cls(
            neighbors_prefix=config['neighbors_prefix'],
            output_prefix=config['output_prefix']
        )

    def learn(self):
        pass

