from typing import Dict, Any, Optional

import networkx as nx
import torch
from torch import Tensor

from agents import TorchAgent
from ml.utils import BIG_NEG


def _get_logprob(neighbors, neighbors_logits):
    inf_tensor = torch.zeros(neighbors_logits.shape)
    inf_tensor[~neighbors.mask] = BIG_NEG
    neighbors_logits = neighbors_logits + inf_tensor
    neighbors_logprobs = torch.nn.functional.log_softmax(neighbors_logits, dim=1)
    return neighbors_logprobs + inf_tensor


class DijkstraAgent(TorchAgent, config_name='dijkstra'):
    def __init__(
            self,
            current_node_idx_prefix: str,
            destination_node_idx_prefix: str,
            neighbors_node_ids_prefix: str,
            topology_prefix: str,
            output_prefix: str,
            bag_ids_prefix: str
    ):
        super().__init__()

        self._current_node_idx_prefix = current_node_idx_prefix
        self._destination_node_idx_prefix = destination_node_idx_prefix
        self._neighbors_node_ids_prefix = neighbors_node_ids_prefix
        self._topology_prefix = topology_prefix
        self._output_prefix = output_prefix
        self._bag_ids_prefix = bag_ids_prefix

    @classmethod
    def create_from_config(cls, config):
        return cls(
            current_node_idx_prefix=config['current_node_idx_prefix'],
            destination_node_idx_prefix=config['destination_node_idx_prefix'],
            neighbors_node_ids_prefix=config['neighbors_node_ids_prefix'],
            topology_prefix=config['topology_prefix'],
            output_prefix=config['output_prefix'],
            bag_ids_prefix=config.get('bag_ids_prefix', None)
        )

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Shape: [batch_size]
        current_node_idx = inputs[self._current_node_idx_prefix]
        batch_size = len(current_node_idx)
        # Shape: [batch_size]
        destination_node_idx = inputs[self._destination_node_idx_prefix]
        # TensorWithMask
        neighbor_node_ids = inputs[self._neighbors_node_ids_prefix]
        # Topology
        topology = inputs[self._topology_prefix]
        try:
            nodes = sorted(topology.graph.nodes)
            path = nx.dijkstra_path(
                topology.graph,
                nodes[current_node_idx.item()],
                nodes[destination_node_idx.item()],
                weight=topology.edge_weight_field
            )
            inputs[self._output_prefix] = torch.tensor([nodes.index(path[1])])
        except:
            inputs[self._output_prefix] = torch.tensor([self._neighbors_node_ids_prefix[0][0]])
        return inputs

    def learn(self) -> Optional[Tensor]:
        return None
