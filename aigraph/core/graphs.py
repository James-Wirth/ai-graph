# aigraph/core/graphs.py
import networkx as nx
from typing import Dict
from aigraph.core.nodes import Node


def graph_from_nodes(nodes: Dict[str, Node]) -> nx.DiGraph:
    G = nx.DiGraph()
    for n in nodes.values():
        G.add_node(n.name, kind="node", node=n)
        for t in n.consumes:
            G.add_node(t, kind="msg")
            G.add_edge(t, n.name)
        for t in n.emits:
            G.add_node(t, kind="msg")
            G.add_edge(n.name, t)
    return G
