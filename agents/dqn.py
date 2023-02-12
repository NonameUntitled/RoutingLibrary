from typing import Dict, Any

from torch import Tensor

from agents import TorchAgent
from agents.base import BaseInputAdapter
from ml.dqn_encoders import BaseQNetwork
from ml.encoders import BaseEncoder
from utils.bag_trajectory import BaseBagTrajectoryMemory


class DqnInputAdapter(BaseInputAdapter, config_name='dqn_input_adapter'):
    def __init__(
            self,
            bag_id,
            node_idx,
            neighbour,
            destination,
            storage
    ):
        self._bag_id = bag_id
        self._node_idx = node_idx
        self._neighbour = neighbour
        self._destination = destination
        self._storage = storage

    @classmethod
    def create_from_config(cls, config):
        return cls(
            bag_id=config.get('bag_id', 'bag_id'),
            node_idx=config.get('node_idx', 'node_idx'),
            neighbour=config.get('neighbour', 'neighbour'),
            destination=config.get('destination', 'destination'),
            storage=config.get('storage', 'storage')
        )

    def get_input(self, inputs: Dict[str, Any]):
        return inputs[self._bag_id], \
               inputs[self._node_idx], \
               inputs[self._neighbour], \
               inputs[self._destination], \
               inputs[self._storage]


class DQNAgent(TorchAgent, config_name='dqn'):
    def __init__(
            self,
            node_idx: int,
            q_network: BaseQNetwork,
            discount_factor: float,
            research_prob: float,
            bag_trajectory_memory: BaseBagTrajectoryMemory,
            dqn_input_adapter: DqnInputAdapter,
            sample_size: int,
    ):
        assert 0 < discount_factor < 1, 'Incorrect `discount_factor` choice'
        assert 0 < research_prob < 1, 'Incorrect `discount_factor` choice'
        super().__init__()
        self._node_idx = node_idx
        self._q_network = q_network
        self._discount_factor = discount_factor
        self._research_prob = research_prob
        self._bag_trajectory_memory = bag_trajectory_memory
        self._dqn_input_adapter = dqn_input_adapter
        self._sample_size = sample_size

    @classmethod
    def create_from_config(cls, config):
        return cls(
            node_idx=config['node_idx'],
            q_network=BaseEncoder.create_from_config(config['q_network']),
            discount_factor=config.get('discount_factor', 0.99),
            research_prob=config.get('research_prob', 0.1),
            bag_trajectory_memory=BaseBagTrajectoryMemory.create_from_config(config['path_memory']),
            dqn_input_adapter=DqnInputAdapter.create_from_config(config['input_adapter']),
            sample_size=config.get('sample_size', 100)
        )

    def forward(self, inputs: Dict[str, Any]) -> Tensor:
        bag_id, node_idx, neighbour, destination, storage = self._ppo_input_adapter.get_input(inputs)
        next_neighbour, _ = self._q_network(
            bag_id=bag_id,
            node_idx=node_idx,
            neighbour=neighbour,
            destination=destination,
            research_prob=self._research_prob,
            storage=storage
        )
        self._bag_trajectory_memory.add_to_trajectory(
            bag_ids=bag_id,
            node_idxs=node_idx,
            extra_infos=zip(bag_id, node_idx, neighbour, destination, storage, next_neighbour)
        )
        return next_neighbour

    def learn(self):
        for trajectory in self._bag_trajectory_memory.sample_trajectories_for_node_idx(
                self._node_idx,
                self._trajectory_sample_size
        ):
            target = trajectory[0]['reward']
            bag_id, node_idx, neighbour, destination, storage, next_neighbour = trajectory[0]['extra_infos']
            if len(trajectory) > 0:
                target += self._discount_factor * (...) # TODO target q network?
            _, neighbour_q = self._q_network(
                bag_id=bag_id,
                node_idx=node_idx,
                neighbour=neighbour,
                destination=destination,
                research_prob=self._research_prob,
                storage=storage
            )
            loss = (target - neighbour_q[next_neighbour]) ** 2
