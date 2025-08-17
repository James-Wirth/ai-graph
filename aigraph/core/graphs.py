import networkx as nx

from typing import Dict, Any
from aigraph.core.agents import Agent

def graph_from_agents(agents: Dict[Any, Agent], start: Any) -> nx.DiGraph:
    g = nx.DiGraph()
    for node, agent in agents.items():
        g.add_node(node, agent=agent)

    if start not in agents:
        raise AssertionError(f"Start node '{start}' not found")

    return g