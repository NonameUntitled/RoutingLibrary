import torch
import numpy as np

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

        neighbors_mask = inputs[f'{self._neighbors_prefix}.mask']  # (batch_size, neighbors_cnt)
        neighbors_mask_sum = torch.sum(neighbors_mask, dim=-1)  # (batch_size)

        next_neighbors = torch.from_numpy(np.random.randint(
            low=np.zeros_like(neighbors_mask_sum.cpu().detach().numpy()),
            high=neighbors_mask_sum.cpu().detach().numpy()
        )).to(neighbors.device)  # (batch_size)

        inputs[self._output_prefix] = next_neighbors  # (batch_size)
        return inputs
