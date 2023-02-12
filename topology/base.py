from utils import MetaParent

from collections import namedtuple
import networkx as nx


class BaseTopology(metaclass=MetaParent):

    @property
    def graph(self):
        raise NotImplementedError

    @property
    def edge_weight_field(self):
        raise NotImplementedError

    @property
    def sink_ids(self):
        raise NotImplementedError


class OrientedTopology(BaseTopology, config_name='oriented'):

    def __init__(self, graph, sink_ids):
        self._graph = graph
        self._sink_ids = sink_ids

    @classmethod
    def create_from_config(cls, config, **kwargs):
        # TODO[Vladimir Baikalov]: I am really lazy to write this one field by field right now
        graph_info = cls._create_graph(
            sources_cfg=config['sources'],
            diverters_cfg=config['diverters'],
            conveyors_cfg=config['conveyors'],
            sink_cfg=config['sinks']
        )
        return cls(**graph_info)

    @staticmethod
    def _create_graph(
            sources_cfg,
            diverters_cfg,
            conveyors_cfg,
            sink_cfg
    ):
        """
        Creates a conveyor network graph from conveyor system layout.
        """
        Section = namedtuple("Section", "type id position")

        # TODO[Vladimir Baikalov]: Add shared IDs for sources, sinks, diverters and junctions

        conveyors_sections = {int(conv_id): [] for conv_id in conveyors_cfg.keys()}

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

                conveyors_sections[int(conveyor_id)].append(Section(
                    type='sink', id=upstream_sink_id, position=length
                ))
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
            'sink_ids': [sink_id for sink_id in sink_cfg]  # TODO[Vladimir Baikalov]: ??? remove list comprehensions
        }

    @property
    def edge_weight_field(self):
        return 'length'

    @property
    def graph(self):
        return self._graph

    @property
    def sink_ids(self):
        return self._sink_ids
