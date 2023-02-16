from utils import MetaParent
from topology.utils import only_reachable_from

from collections import namedtuple, defaultdict
import networkx as nx
import numpy as np
import random


class BaseTopology(metaclass=MetaParent):

    @property
    def schema(self):
        raise NotImplementedError

    @property
    def graph(self):
        raise NotImplementedError

    @property
    def edge_weight_field(self):
        raise NotImplementedError

    @property
    def sinks(self):
        raise NotImplementedError


class OrientedTopology(BaseTopology, config_name='oriented'):

    def __init__(self, graph, sinks):
        self._graph = graph
        self._sinks = sinks

    @classmethod
    def create_from_config(cls, config):
        # TODO[Vladimir Baikalov]: I am really lazy to write this one field by field right now
        graph_info = cls._create_graph(
            sources_cfg=config['sources'],
            diverters_cfg=config['diverters'],
            conveyors_cfg=config['conveyors']
        )
        return cls(**graph_info)

    @staticmethod
    def _create_graph(
            sources_cfg,
            diverters_cfg,
            conveyors_cfg
    ):
        """
        Creates a conveyor network graph from conveyor system layout.
        """
        Section = namedtuple("Section", "type id position")

        # TODO[Vladimir Baikalov]: Add shared IDs for sources, sinks, diverters and junctions

        conveyors_sections = {int(conv_id): [] for conv_id in conveyors_cfg.keys()}
        sinks = []

        for source_node_id, source_node_cfg in sources_cfg.items():
            upstream_conveyor_id = source_node_cfg['upstream_conv']
            conveyors_sections[upstream_conveyor_id].append(Section(
                type='source', id=int(source_node_id), position=0
            ))

        for diverter_node_id, diverter_cfg in diverters_cfg.items():
            conveyor_id = diverter_cfg['conveyor']
            conveyor_position = diverter_cfg['pos']
            upstream_conveyor_id = diverter_cfg['upstream_conv']

            conveyors_sections[conveyor_id].append(Section(
                type='diverter', id=int(diverter_node_id), position=conveyor_position
            ))

            conveyors_sections[upstream_conveyor_id].append(Section(
                type='diverter', id=int(diverter_node_id), position=0
            ))

        junction_idx = 0
        for conveyor_id, conveyor_cfg in conveyors_cfg.items():
            length = conveyor_cfg['length']
            upstream_cfg = conveyor_cfg['upstream']

            if upstream_cfg['type'] == 'sink':
                upstream_sink_id = upstream_cfg['id']

                sink_node = Section(
                    type='sink', id=upstream_sink_id, position=length
                )
                sinks.append(sink_node)

                conveyors_sections[int(conveyor_id)].append(sink_node)
            elif upstream_cfg['type'] == 'conveyor':
                upstream_conveyor_id = upstream_cfg['id']
                upstream_conveyor_position = upstream_cfg['pos']

                conveyors_sections[int(conveyor_id)].append(Section(
                    type='junction', id=junction_idx, position=length
                ))

                conveyors_sections[upstream_conveyor_id].append(Section(
                    type='junction', id=junction_idx, position=upstream_conveyor_position
                ))

                junction_idx += 1
            else:
                raise Exception('Invalid conveyor upstream type: ' + upstream_cfg['type'])

        graph = nx.DiGraph()

        for conveyor_id, conveyor_section in conveyors_sections.items():
            conveyor_section = sorted(conveyor_section, key=lambda section: section.position)
            assert conveyor_section[0].position == 0, \
                f'No node at the beginning of conveyor {conveyor_id}!'
            assert conveyor_section[-1].position == conveyors_cfg[str(conveyor_id)]['length'], \
                f'No node at the end of conveyor {conveyor_id}!'

            for i in range(1, len(conveyor_section)):
                fst_node_cfg = conveyor_section[i - 1]
                snd_node_cfg = conveyor_section[i]

                fst_node_position = fst_node_cfg.position
                snd_node_position = snd_node_cfg.position

                edge_length = snd_node_position - fst_node_position

                assert edge_length >= 2, \
                    f'Conveyor section of conveyor {conveyor_id} is too short. ' \
                    f'Positions: {fst_node_position} and {fst_node_position}!'

                graph.add_edge(
                    fst_node_cfg,
                    snd_node_cfg,
                    length=edge_length,
                    conveyor=conveyor_id,
                    end_pos=snd_node_position
                )

                if (
                        i > 1  # This node stays on this conveyor (at some point)
                        or fst_node_cfg.type != 'diverter'  # This conveyor goes from source
                ):
                    graph.nodes[fst_node_cfg]['conveyor'] = conveyor_id
                    graph.nodes[fst_node_cfg]['conveyor_pos'] = fst_node_position

        return {
            'graph': graph,
            'sinks': sinks
        }

    def gen_episodes(self, num_samples):
        graph = self.graph
        graph_nodes = sorted(graph.nodes)
        sinks = self.sinks

        adjacency_matrix = nx.convert_matrix.to_numpy_array(
            graph,
            nodelist=graph_nodes,
            weight='length',
            dtype=np.float32
        )

        # Create mapping from node to idx
        node_2_idx = {
            node: idx for idx, node in enumerate(graph_nodes)
        }

        # Sanity check that paths are correct TODO[Vladimir Baikalov]: check correctness and then remove
        for from_idx, from_node in enumerate(graph_nodes):
            for to_idx, to_node in enumerate(graph_nodes):
                if adjacency_matrix[from_idx][to_idx] > 0:  # edge exists
                    assert nx.path_weight(
                        graph,
                        [from_node, to_node],
                        weight='length'
                    ) == adjacency_matrix[from_idx][to_idx]

        best_transitions = defaultdict(dict)  # start_node, finish_node -> best_node_to_go_into
        path_lengths = defaultdict(dict)  # start_node, finish_node -> path_length

        # Shortest path preprocessing
        for start_node in graph_nodes:
            for finish_node in graph_nodes:
                if start_node != finish_node and nx.has_path(graph, start_node, finish_node):
                    path = nx.dijkstra_path(graph, start_node, finish_node, weight='length')
                    length = nx.path_weight(graph, path, weight='length')
                    best_transitions[start_node][finish_node] = path[1] if len(path) > 1 else start_node
                    path_lengths[start_node][finish_node] = length

        dataset = []
        # TODO[Vladimir Baikalov]: try usage of augmented graph for pre-train later
        while len(dataset) < num_samples:
            destination_node = random.choice(sinks)

            current_node = destination_node
            while current_node == destination_node:
                current_node = random.choice(only_reachable_from(
                    graph=graph,
                    final_node=destination_node,
                    start_nodes=graph_nodes
                ))

            out_neighbors = graph.successors(current_node)
            filtered_out_neighbors = only_reachable_from(
                graph=graph,
                final_node=destination_node,
                start_nodes=out_neighbors
            )  # TODO[Vladimir Baikalov]: maybe remove filtering

            if len(filtered_out_neighbors) == 0:
                continue

            # Add basic information
            sample = {
                'current_node_idx': node_2_idx[current_node],
                'neighbors_node_ids': [
                    node_2_idx[filtered_out_neighbor]
                    for filtered_out_neighbor in filtered_out_neighbors
                ],
                'destination_node_idx': node_2_idx[destination_node]
            }

            # Add algorithm-specific information
            # TODO[Vladimir Baikalov]: Generalize it
            # dqn-specific
            path_lengths = []
            for filtered_out_neighbor in filtered_out_neighbors:
                path = nx.dijkstra_path(
                    graph,
                    filtered_out_neighbor,
                    destination_node,
                    weight='length'
                )
                # In the line below we append negative value because we want to minimize actual path
                path_lengths.append(nx.path_weight(graph, path, weight='length'))
            sample['path_lengths'] = path_lengths

            # ppo-specific
            path = nx.dijkstra_path(
                graph,
                current_node,
                destination_node,
                weight='length'
            )  # List of nodes in the path from `current_node` to `destination_node`

            best_transition_node = path[1]
            for neighbor_idx, filtered_out_neighbor in enumerate(filtered_out_neighbors):
                if filtered_out_neighbor == best_transition_node:
                    sample['next_node_idx'] = neighbor_idx
                    break
            else:
                assert False, "There is no such neighbor!"

            # Here we want to store negative path because in future we would like to minimize it
            sample['path_length'] = -nx.path_weight(graph, path, weight='length')

            dataset.append(sample)

        return dataset

    @property
    def schema(self):
        return {
            'current_node_idx': {
                'type': 'long',
                'is_ragged': False
            },
            'neighbors_node_ids': {
                'type': 'long',
                'is_ragged': True
            },
            'destination_node_idx': {
                'type': 'long',
                'is_ragged': False
            },
            'path_lengths': {
                'type': 'float',
                'is_ragged': True
            },
            'next_node_idx': {
                'type': 'long',
                'is_ragged': False
            },
            'path_length': {
                'type': 'float',
                'is_ragged': False
            }
        }

    @property
    def edge_weight_field(self):
        return 'length'

    @property
    def graph(self):
        return self._graph

    @property
    def sinks(self):
        return self._sinks