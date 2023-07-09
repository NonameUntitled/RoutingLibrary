import copy
from typing import Dict, Any, Callable, Optional

import torch
from torch import Tensor, nn

import utils
from agents import TorchAgent
from ml import BaseOptimizer
from ml.dqn_encoders import BaseQNetwork
from ml.encoders import BaseEncoder
from ml.utils import BIG_NEG, TensorWithMask
from topology.utils import only_reachable_from
from utils.bag_trajectory import BaseBagTrajectoryMemory


class DQNAgent(TorchAgent, config_name='dqn'):
    def __init__(
            self,
            current_node_idx_prefix: str,
            destination_node_idx_prefix: str,
            neighbors_node_ids_prefix: str,
            output_prefix: str,
            bag_ids_prefix: str,
            q_network: BaseQNetwork,
            discount_factor: float = 0.99,
            research_prob: float = 0.1,
            bag_trajectory_memory: BaseBagTrajectoryMemory = None,
            trajectory_sample_size: int = 30,
            optimizer_factory: Callable[[nn.Module], BaseOptimizer] = None
    ):
        assert 0 < discount_factor < 1, 'Incorrect `discount_factor` choice'
        assert 0 < research_prob < 1, 'Incorrect `discount_factor` choice'
        super().__init__()

        self._current_node_idx_prefix = current_node_idx_prefix
        self._destination_node_idx_prefix = destination_node_idx_prefix
        self._neighbors_node_ids_prefix = neighbors_node_ids_prefix
        self._output_prefix = output_prefix
        self._bag_ids_prefix = bag_ids_prefix

        self._q_network = q_network
        self._discount_factor = discount_factor
        self._research_prob = research_prob
        self._bag_trajectory_memory = bag_trajectory_memory
        self._trajectory_sample_size = trajectory_sample_size

        self._optimizer = optimizer_factory(self) if optimizer_factory is not None else None
        self._optimizer_factory = optimizer_factory

    @classmethod
    def create_from_config(cls, config):
        return cls(
            current_node_idx_prefix=config['current_node_idx_prefix'],
            destination_node_idx_prefix=config['destination_node_idx_prefix'],
            neighbors_node_ids_prefix=config['neighbors_node_ids_prefix'],
            output_prefix=config['output_prefix'],
            bag_ids_prefix=config.get('bag_ids_prefix', None),
            q_network=BaseEncoder.create_from_config(config['q_network']),
            discount_factor=config.get('discount_factor', 0.99),
            research_prob=config.get('research_prob', 0.1),
            bag_trajectory_memory=BaseBagTrajectoryMemory.create_from_config(config['path_memory'])
            if 'path_memory' in config else None,
            trajectory_sample_size=config.get('trajectory_sample_size', 30),
            optimizer_factory=lambda m: BaseOptimizer.create_from_config(config['optimizer'], model=m)
            if 'optimizer' in config else None
        )

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Shape: [batch_size]
        current_node_idx = inputs[self._current_node_idx_prefix]
        batch_size = len(current_node_idx)
        # Shape: [batch_size]
        destination_node_idx = inputs[self._destination_node_idx_prefix]
        # TensorWithMask
        # Shape: [batch_size, max_neighbors_num]
        neighbor_node_ids = inputs[self._neighbors_node_ids_prefix]

        # Shape: [batch_size], [batch_size, max_neighbors_num]
        next_neighbor_ids, neighbors_q = self._q_network(
            current_node_idx=current_node_idx,
            neighbor_node_ids=neighbor_node_ids,
            destination_node_idx=destination_node_idx
        )

        if self._bag_trajectory_memory is not None:
            # TODO[Vladimir Baikalov]: Think about how to generalize
            bag_ids = inputs.get(self._bag_ids_prefix, None)

            self._bag_trajectory_memory.add_to_trajectory(
                bag_ids=bag_ids,
                node_idxs=torch.full([batch_size], self.node_id),
                extra_infos=zip(
                    torch.unsqueeze(current_node_idx, dim=1),
                    neighbor_node_ids,
                    neighbors_q,
                    torch.unsqueeze(destination_node_idx, dim=1),
                    torch.unsqueeze(next_neighbor_ids, dim=1)
                )
            )
        inputs[self._output_prefix] = next_neighbor_ids
        # TODO[Zhogov Alexandr] fix it
        flatten_neighbors_q = neighbors_q.flatten()
        inputs.update({
            'predicted_next_node_idx': next_neighbor_ids,
            'predicted_next_node_q': flatten_neighbors_q[flatten_neighbors_q != BIG_NEG],
        })
        return inputs

    def learn(self) -> Optional[Tensor]:
        loss = 0
        # We consider only one step because it was proposed directly in the original paper
        learn_trajectories = self._bag_trajectory_memory.sample_trajectories_for_node_idx(
            node_idx=self.node_id,
            count=self._trajectory_sample_size,
            length=2
        )
        if not learn_trajectories:
            return None
        for trajectory in learn_trajectories:
            rewards = [reward for _, reward in trajectory]
            parts = [part for part, _ in trajectory]
            target = rewards[0]
            current_node_idx, neighbor_node_ids, _, destination_node_idx, next_neighbor = parts[0].extra_info
            if len(parts) > 1:
                _, neighbor_node_ids_, neighbor_q_, _, next_neighbor_ = parts[1].extra_info
                neighbor_q_ = torch.unsqueeze(neighbor_q_, dim=0)
                target += self._discount_factor * neighbor_q_[
                    neighbor_node_ids_.padded_values == next_neighbor_].detach()
            _, neighbor_q = self._q_network(
                current_node_idx=current_node_idx,
                neighbor_node_ids=neighbor_node_ids,
                destination_node_idx=destination_node_idx
            )
            # TODO[Vladimir Baikalov] check
            loss += (target - neighbor_q[neighbor_node_ids.padded_values == next_neighbor]) ** 2
        loss /= len(learn_trajectories)
        self._optimizer.step(loss)
        return loss.detach().item()

    def debug(self, topology, step_num):
        nodes_list = sorted(topology.graph.nodes)
        nodes = {node: idx for idx, node in enumerate(nodes_list)}
        sinks = list(filter(lambda n: n.type == 'sink', topology.graph.nodes))
        diverter = nodes_list[self.node_id]
        for sink in sinks:
            all_neighbors = list(topology.graph.successors(diverter))
            neighbors = only_reachable_from(
                graph=topology.graph,
                final_node=sink,
                start_nodes=all_neighbors
            )
            if not neighbors:
                continue
            _, q_func = self._q_network(
                current_node_idx=torch.LongTensor([nodes[diverter]]),
                neighbor_node_ids=TensorWithMask(torch.LongTensor([nodes[n] for n in neighbors]), torch.LongTensor([len(neighbors)])),
                destination_node_idx=torch.LongTensor([nodes[sink]])
             )
            for idx, q in enumerate(q_func[0]):
                if utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER:
                    utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.add_scalar(
                        '{}/{}'.format('train', f'q_func_{diverter}_{neighbors[idx]}_{sink}'),
                        q.item(),
                        step_num
                      )
                    utils.tensorboard_writers.GLOBAL_TENSORBOARD_WRITER.flush()

    def _copy(self):
        agent_copy = copy.deepcopy(self)
        agent_copy._optimizer = agent_copy._optimizer_factory(agent_copy)
        return agent_copy
