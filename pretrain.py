from utils import create_logger, fix_random_seed, parse_args
from utils import MetaParent
from utils import DEVICE

from agents import BaseAgent
from ml import BaseLoss
from topology import BaseTopology

from collections import defaultdict
import networkx as nx
import numpy as np
import random

from torch.utils.data import DataLoader

logger = create_logger(name=__name__)
seed_val = 42


class BasePretrainRunner(metaclass=MetaParent):

    def run(self):
        raise NotImplementedError


class ConveyorPretrainRunner(BasePretrainRunner, config_name='conveyor'):

    def __init__(
            self,
            agent,
            topology,
            num_epochs,
            num_samples,
            filename=None
    ):
        self._agent = agent
        self._topology = topology
        self._num_epochs = num_epochs
        self._num_samples = num_samples
        self._filename = filename

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            agent=BaseAgent.create_from_config(config['agent']),
            topology=BaseTopology.create_from_config(config['topology']),
            num_epochs=config['num_epochs'],
            num_samples=config['samples_count'],
            filename=config.get('filename', None)
        )

    def run(self, model, loss_function, optimizer, callback):
        dataset = self._gen_episodes()
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=64,  # TODO fix
            shuffle=True,
            drop_last=True
        )

        step_num = 0
        for epoch in range(self._num_epochs):
            logger.debug(f'Start pre-train epoch {epoch}')
            for step, batch in enumerate(dataloader):
                model.train()

                for key, values in batch.items():
                    batch[key] = batch[key].to(DEVICE)

                model_output = model(batch)
                batch.update(model_output)

                loss = loss_function(batch)

                optimizer.step(loss)
                callback(batch, step_num)
                step_num += 1

        logger.debug('Start pre-training...')

        logger.debug('Training procedure has been finished!')
        return model.state_dict()

    def _gen_episodes(self):
        graph = self._topology.graph
        graph_nodes = sorted(graph.nodes)
        sinks_ids = self._topology.sinks_ids

        adjacency_matrix = nx.convert_matrix.to_numpy_array(
            graph,
            nodelist=graph_nodes,
            weight='length',
            dtype=np.float32
        )

        # Sanity check that paths are correct TODO[Vladimir Baikalov]: check correctness and then remove
        edges_indexes = []  # ???
        for from_idx, from_node in enumerate(graph_nodes):
            for to_idx, to_node in enumerate(graph_nodes):
                if adjacency_matrix[from_idx][to_idx] > 0:  # edge exists
                    edges_indexes.append((from_node, to_node))
                    assert nx.path_weight(
                        graph,
                        [from_node, to_node],
                        weight='length'
                    ) == adjacency_matrix[from_idx][to_idx]

        best_transitions = defaultdict(dict)  # start_node, finish_node -> best_node_to_go_into
        path_lengths = defaultdict(dict)  # start_node, finish_node -> path_le

        # Shortest path preprocessing
        for start_node in graph_nodes:
            for finish_node in graph_nodes:
                if start_node != finish_node and nx.has_path(graph_nodes, start_node, finish_node):
                    path = nx.dijkstra_path(graph, start_node, finish_node, weight='length')
                    length = nx.path_weight(graph, path, weight='length')
                    best_transitions[start_node][finish_node] = path[1] if len(path) > 1 else start_node
                    path_lengths[start_node][finish_node] = length

        dataset = []
        # TODO[Vladimir Baikalov]: try usage of augmented graph for pre-train later
        while len(dataset) < self._num_samples:
            destination_node_idx = random.choice(sinks_ids)

            current_node_idx = destination_node_idx
            while current_node_idx == destination_node_idx:
                current_node_idx = random.choice(only_reachable(
                    graph,
                    destination_node_idx,
                    graph_nodes
                ))

            out_neighbors_ids = graph.successors(current_node_idx)
            filtered_out_neighbors_ids = only_reachable(
                graph,
                destination_node_idx,
                out_neighbors_ids
            )  # TODO[Vladimir Baikalov]: maybe remove filtering

            if len(filtered_out_neighbors_ids) == 0:
                continue

            # Add basic information
            sample = {
                'current_node_idx': current_node_idx,
                'neighbors_node_ids': out_neighbors_ids,
                'destination_node_idx': destination_node_idx
            }

            # Add algorithm-specific information
            # TODO[Vladimir Baikalov]: Generalize it
            # dqn-specific
            path_lengths = []
            for out_neighbors_idx in out_neighbors_ids:
                path = nx.dijkstra_path(
                    graph,
                    out_neighbors_idx,
                    destination_node_idx,
                    weight='length'
                )
                path_lengths.append(nx.path_weight(graph, path, weight='length'))
            sample['path_lengths'] = path_lengths

            # ppo-specific
            path = nx.dijkstra_path(
                graph,
                current_node_idx,
                destination_node_idx,
                weight='length'
            )  # List of nodes in the path from `current_node_idx` to `destination_node_idx`
            sample['next_node_idx'] = path[1]  # TODO[Vladimir Baikalov]: Check that this is correct
            sample['path_length'] = nx.path_weight(graph, path, weight='length')

            dataset.append(sample)

        return dataset


def main():
    params = parse_args()
    fix_random_seed(seed_val)

    # Environment-related part initialization
    topology = BaseTopology.create_from_config(params['topology'])

    # Ml-related part initialization
    # agent = BaseAgent.create_from_config(params['agent'])
    loss_function = BaseLoss.create_from_config(params['loss'])
    # optimizer = BaseOptimizer.create_from_config(params['optimizer'], model=agent)
    # callback = BaseCallback.create_from_config(params['callback'], model=agent, optimizer=optimizer)

    # Pretrain pipeline initialization
    # pretrain_runner = BasePretrainRunner.create_from_config(
    #     params['pretrain_params'],
    #     topology=topology
    # )

    # TODO[Vladimir Baikalov]: Move somewhere
    # save_path = os.path.join(CHECKPOINT_DIR, params['model_name'])  # dir_with_models and pretrain_filename

    # pretrain_runner.run(
    #     model=agent,
    #     loss_function=loss_function,
    #     optimizer=optimizer,
    #     callback=callback
    # )


if __name__ == '__main__':
    main()
