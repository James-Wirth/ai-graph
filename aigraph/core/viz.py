from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from aigraph.core.agents import Agent, LLMAgent


def _compute_layout(G: nx.DiGraph) -> Dict[Any, Tuple[float, float]]:
    try:
        from networkx.drawing.nx_agraph import graphviz_layout 
        return graphviz_layout(G, prog="dot")
    except Exception:
        pass
    try:
        from networkx.drawing.nx_pydot import graphviz_layout as pydot_layout  
        return pydot_layout(G, prog="dot")
    except Exception:
        pass
    return nx.spring_layout(G, seed=42)


def _path_edges_from_history(history: List[Dict[str, Any]]) -> List[Tuple[Any, Any]]:
    path_edges: List[Tuple[Any, Any]] = []
    for step in history:
        node = step.get("node")
        decision = step.get("decision") or {}
        nxt = decision.get("next_node")
        if node is not None and nxt is not None:
            path_edges.append((node, nxt))
    return path_edges


def render_workflow_graph(
    G: nx.DiGraph,
    *,
    start: Optional[Any] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    figsize: Tuple[int, int] = (10, 6),
    filename: Optional[str] = None,
    title: Optional[str] = None,
):
    routers: List[Any] = []
    terminals: List[Any] = []
    normals: List[Any] = []
    tool_capable: set[Any] = set()

    for n, data in G.nodes(data=True):
        agent = data.get("agent")
        if not isinstance(agent, Agent):
            raise AssertionError(f"Node '{n}' missing a valid Agent (found: {type(agent)})")

        is_router = isinstance(agent, LLMAgent) and bool(getattr(agent, "edges", {}))
        is_terminal = G.out_degree(n) == 0

        node_def = data.get("node_def")
        has_declared_tools = bool(getattr(node_def, "tools", None))
        has_agent_tools = bool(isinstance(agent, LLMAgent) and getattr(agent, "allowed_tools", None))
        if has_declared_tools or has_agent_tools:
            tool_capable.add(n)
        
        if is_terminal:
            terminals.append(n)
        elif is_router:
            routers.append(n)
        else:
            normals.append(n)

    pos = _compute_layout(G)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_axis_off()

    all_edges = list(G.edges())
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=all_edges,
        width=1.2,
        edge_color="#9aa0a6",
        arrows=True,
        arrowsize=16,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.05",
        ax=ax,
    )

    path_edges: List[Tuple[Any, Any]] = _path_edges_from_history(history) if history else []
    if path_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=path_edges,
            width=3.0,
            edge_color="#d62728", 
            arrows=True,
            arrowsize=18,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.05",
            ax=ax,
        )

    def _draw_nodes(nodes: Iterable[Any], shape: str, facecolor: str):
        if not nodes:
            return

        edgecolors = [("#ff7f0e" if n in tool_capable else "#202124") for n in nodes]  
        linewidths = [(2.5 if n in tool_capable else 1.2) for n in nodes]
   
        for i, n in enumerate(nodes):
            if start is not None and n == start:
                edgecolors[i] = "#2ca02c"  
                linewidths[i] = 3.0

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(nodes),
            node_size=1400,
            node_shape=shape,
            node_color=facecolor,
            edgecolors=edgecolors,
            linewidths=linewidths,
            ax=ax,
        )

    _draw_nodes(normals, shape="o", facecolor="#e8eaed")  
    _draw_nodes(routers, shape="s", facecolor="#d2e3fc")   
    _draw_nodes(terminals, shape="D", facecolor="#fce8e6") 

    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax)

    legend_elems = [
        Patch(facecolor="#d2e3fc", edgecolor="#202124", label="Routing node"),
        Patch(facecolor="#e8eaed", edgecolor="#202124", label="Normal node"),
        Patch(facecolor="#fce8e6", edgecolor="#202124", label="End node"),
        Line2D([0], [0], color="#d62728", lw=3, label="Path taken"),
        Line2D([0], [0], color="#ff7f0e", lw=2.5, label="Tool-capable"),
        Line2D([0], [0], color="#2ca02c", lw=3, label="Start node"),
    ]
    ax.legend(
        handles=legend_elems,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False
    )

    if title:
        ax.set_title(title)

    if filename:
        fig.savefig(filename, bbox_inches="tight")
    return fig, ax


def mermaid_from_graph(
    G: nx.DiGraph,
    *,
    start: Optional[Any] = None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> str:

    lines: List[str] = ["flowchart LR"]

    for n, data in G.nodes(data=True):
        agent = data.get("agent")
        label = str(n)
        shape_open, shape_close = ("([", "])")  

        is_terminal = G.out_degree(n) == 0
        is_router = isinstance(agent, LLMAgent) and bool(getattr(agent, "edges", {}))
        if is_terminal:
            shape_open, shape_close = ("{{", "}}")  
        elif is_router:
            shape_open, shape_close = ("[", "]")    
        if start is not None and n == start:
            label = f"{label} ✦"
        lines.append(f'  {n}{shape_open}{label}{shape_close}')

    path_set = set(_path_edges_from_history(history) if history else [])
    for u, v in G.edges():
        style = "-->" if (u, v) not in path_set else "===>"
        lines.append(f"  {u} {style} {v}")

    return "\n".join(lines)
