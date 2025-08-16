import networkx as nx

from typing import Dict, Any
from aigraph.core.agents import Agent, LLMAgent

def graph_from_agents(agents: Dict[Any, Agent], start: Any) -> nx.DiGraph:
    g = nx.DiGraph()
    for node, agent in agents.items():
        g.add_node(node, agent=agent)

    for node, agent in agents.items():
        if isinstance(agent, LLMAgent) and agent.edges:
            for _, dst in agent.edges.items():
                if dst not in agents:
                    raise AssertionError(f"Agent '{agent.name}' chose edge to unknown node '{dst}'")
                g.add_edge(node, dst)

    if start not in agents:
        raise AssertionError(f"Start node '{start}' not found")

    return g
