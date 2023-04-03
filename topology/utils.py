import networkx as nx
from collections import namedtuple

Section = namedtuple("Section", "type id position")


def only_reachable_from(graph, final_node, start_nodes):
    return list(filter(lambda node: nx.has_path(graph, node, final_node), start_nodes))


def node_type(nid: Section) -> str:
    assert isinstance(nid, Section), "node_type: nid must be a Section"
    return nid.type


def conveyor_edges(graph: nx.DiGraph, conv_idx: int) -> list[tuple[Section, Section]]:
    edges = [(u, v) for u, v, cid in graph.edges(data='conveyor')
             if cid == conv_idx]
    return sorted(edges, key=lambda e: graph[e[0]][e[1]]['end_pos'])


def conveyor_adj_nodes(graph: nx.DiGraph, conv_idx: int, only_own=False) -> list[Section]:
    conv_edges = conveyor_edges(graph, conv_idx)

    nodes = [conv_edges[0][0]]
    for _, v in conv_edges:
        nodes.append(v)

    if only_own:
        if node_type(nodes[0]) != 'junction':
            nodes.pop(0)
        nodes.pop()

    return nodes


def conveyor_adj_nodes_with_data(graph: nx.DiGraph, conv_idx: int, data: str, only_own=False) -> list[dict[str, int | Section]]:
    conv_edges = conveyor_edges(graph, conv_idx)

    nodes = [conv_edges[0][0]]
    for _, v in conv_edges:
        nodes.append(v)

    if only_own:
        if node_type(nodes[0]) != 'junction':
            nodes.pop(0)
        nodes.pop()

    nodes = [{"node": n, "position": graph.nodes[n][data]} for n in nodes]
    return nodes


def node_id(nid: Section) -> int:
    assert isinstance(nid, Section), "node_id: nid must be a Section"
    return nid.id


def conveyor_idx(graph: nx.DiGraph, node: Section) -> int:
    ntype = node_type(node)
    rwgwg = graph.nodes[node]
    if ntype == 'conveyor':
        return node_id(node)
    elif ntype == 'sink':
        return -1
    else:
        return graph.nodes[node]['conveyor']


def node_conv_pos(graph: nx.DiGraph, conv_idx: int, node: Section) -> int | None:
    es = conveyor_edges(graph, conv_idx)
    p_pos = 0
    for u, v in es:
        if u.type == node.type and u.id == node.id:
            return p_pos
        p_pos = graph[u][v]['end_pos']
        if v.type == node.type and v.id == node.id:
            return p_pos
    return None

def conv_start_node(graph: nx.DiGraph, conv_idx: int) -> Section:
    es = conveyor_edges(graph, conv_idx)
    return es[0][0]

def conv_next_node(graph: nx.DiGraph, conv_idx: int, node: Section) -> Section:
    es = conveyor_edges(graph, conv_idx)
    for u, v in es:
        if u.type == node.type and u.id == node.id:
            return v

def get_node_by_id(topology, node_id):
    for node in topology._node_2_idx.keys():
        if topology._node_2_idx[node] == node_id:
            return node
    return None
