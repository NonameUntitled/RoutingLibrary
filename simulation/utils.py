import networkx as nx

from typing import Tuple, TypeVar, Union, List, Callable, Iterable

AgentId = Tuple[str, int]


def agent_type(aid):
    if type(aid) == tuple:
        return aid[0]
    return aid


def agent_idx(aid):
    if type(aid) == tuple:
        return aid[1]
    return aid


def make_conveyor_topology_graph(config) -> nx.DiGraph:
    """
    Creates a conveyor network graph from conveyor system layout.
    """
    sources = config['sources']
    conveyors = config['conveyors']
    diverters = config['diverters']

    conv_sections = {conv_id: [] for conv_id in range(0, len(conveyors))}

    for src_id, src in enumerate(sources):
        conv = src['upstream_conv']
        conv_sections[conv].append(('source', src_id, 0))

    for dv_id, dv in enumerate(diverters):
        conv = dv['conveyor']
        pos = dv['pos']
        up_conv = dv['upstream_conv']
        conv_sections[conv].append(('diverter', dv_id, pos))
        conv_sections[up_conv].append(('diverter', dv_id, 0))

    junction_idx = 0
    for conv_id, conv in enumerate(conveyors):
        l = conv['length']
        up = conv['upstream']

        if up['type'] == 'sink':
            conv_sections[conv_id].append(('sink', up['idx'], l))
        elif up['type'] == 'conveyor':
            up_id = up['idx']
            up_pos = up['pos']
            conv_sections[conv_id].append(('junction', junction_idx, l))
            conv_sections[up_id].append(('junction', junction_idx, up_pos))
            junction_idx += 1
        else:
            raise Exception('Invalid conveyor upstream type: ' + up['type'])

    DG = nx.DiGraph()
    for (conv_id, sections) in conv_sections.items():
        sections.sort(key=lambda s: s[2])
        assert sections[0][2] == 0, \
            f"No node at the beginning of conveyor {conv_id}!"
        assert sections[-1][2] == conveyors[conv_id]['length'], \
            f"No node at the end of conveyor {conv_id}!"

        for i in range(1, len(sections)):
            u = sections[i - 1][:-1]
            v = sections[i][:-1]
            u_pos = sections[i - 1][-1]
            v_pos = sections[i][-1]
            edge_len = v_pos - u_pos

            assert edge_len >= 2, f"Conveyor section of conveyor {conv_id} is way too short (positions: {u_pos} and {v_pos})!"
            DG.add_edge(u, v, length=edge_len, conveyor=conv_id, end_pos=v_pos)

            if (i > 1) or (u[0] != 'diverter'):
                DG.nodes[u]['conveyor'] = conv_id
                DG.nodes[u]['conveyor_pos'] = u_pos

    return DG


def conveyor_idx(topology, node):
    atype = agent_type(node)
    if atype == 'conveyor':
        return agent_idx(node)
    elif atype == 'sink':
        return -1
    else:
        return topology.nodes[node]['conveyor']


def node_conv_pos(topology, conv_idx, node):
    es = conveyor_edges(topology, conv_idx)
    p_pos = 0
    for u, v in es:
        if u == node:
            return p_pos
        p_pos = topology[u][v]['end_pos']
        if v == node:
            return p_pos
    return None


def conveyor_edges(topology, conv_idx):
    edges = [(u, v) for u, v, cid in topology.edges(data='conveyor')
             if cid == conv_idx]
    return sorted(edges, key=lambda e: topology[e[0]][e[1]]['end_pos'])


def conveyor_adj_nodes(topology, conv_idx, only_own=False, data=False):
    conv_edges = conveyor_edges(topology, conv_idx)

    nodes = [conv_edges[0][0]]
    for _, v in conv_edges:
        nodes.append(v)

    if only_own:
        if agent_type(nodes[0]) != 'junction':
            nodes.pop(0)
        nodes.pop()

    if data:
        nodes = [(n, (topology.nodes[n][data] if type(data) != bool else topology.nodes[n]))
                 for n in nodes]
    return nodes


T = TypeVar('T')
X = TypeVar('X')


def binary_search(ls: List[T], diff_func: Callable[[T], X],
                  return_index: bool = False,
                  preference: str = 'nearest') -> Union[Tuple[T, int], T, None]:
    """
    Binary search via predicate.
    preference param:
    - 'nearest': with smallest diff, result always exists in non-empty list
    - 'next': strictly larger
    - 'prev': strictly smaller
    """
    if preference not in ('nearest', 'next', 'prev'):
        raise ValueError('binary search: invalid preference: ' + preference)

    if len(ls) == 0:
        return None

    l = 0
    r = len(ls)
    while l < r:
        m = l + (r - l) // 2
        cmp_res = diff_func(ls[m])
        if cmp_res == 0:
            return (ls[m], m) if return_index else ls[m]
        elif cmp_res < 0:
            r = m
        else:
            l = m + 1

    if l >= len(ls):
        l -= 1

    if (preference == 'nearest') and (l > 0) and (abs(diff_func(ls[l - 1])) < abs(diff_func(ls[l]))):
        l -= 1
    elif (preference == 'prev') and (diff_func(ls[l]) < 0):
        if l > 0:
            l -= 1
        else:
            return None
    elif (preference == 'next') and (diff_func(ls[l]) > 0):
        if l < len(ls) - 1:
            l += 1
        else:
            return None

    return (ls[l], l) if return_index else ls[l]


def differs_from(x: T, using=None) -> Callable[[T], X]:
    def _diff(y):
        if using is not None:
            y = using(y)
        return x - y

    return _diff


def merge_sorted(list_a, list_b, using):
    if len(list_a) == 0:
        return list_b
    if len(list_b) == 0:
        return list_a
    i = 0
    j = 0
    res = []
    while i < len(list_a) and j < len(list_b):
        if using(list_a[i]) < using(list_b[j]):
            res.append(list_a[i])
            i += 1
        else:
            res.append(list_b[j])
            j += 1
    return res + list_a[i:] + list_b[j:]


def def_list(ls, default=[]):
    if ls is None:
        return list(default)
    elif isinstance(ls, Iterable) and not (type(ls) == str):
        return list(ls)
    return [ls]
