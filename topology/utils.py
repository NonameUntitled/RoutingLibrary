import networkx as nx


def only_reachable_from(graph, final_node, start_nodes):
    return list(filter(lambda node: nx.has_path(graph, node, final_node), start_nodes))