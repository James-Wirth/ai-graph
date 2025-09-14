# aigraph/core/viz.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Set, DefaultDict
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _observed_node_edges(history: Optional[List[Dict[str, Any]]]) -> Set[Tuple[str, str]]:
    if not history:
        return set()
    last_emitter: Dict[str, Optional[str]] = {}
    observed: Set[Tuple[str, str]] = set()

    for ev in history:
        node = ev.get("node")
        consumed = ev.get("consumed")
        emitted = ev.get("emitted") or []

        if node and consumed:
            src = last_emitter.get(consumed)
            if src and src != node:
                observed.add((src, node))

        if node:
            for t in emitted:
                last_emitter[t] = node

    return observed


def _declared_edges(
    G: nx.DiGraph,
    history: Optional[List[Dict[str, Any]]],
) -> Tuple[Set[str], Set[Tuple[str, str]], Dict[str, str]]:
    nodes: Set[str] = set()
    node_labels: Dict[str, str] = {}
    producers: DefaultDict[str, Set[str]] = defaultdict(set)
    consumers: Dict[str, str] = {}

    for n, data in G.nodes(data=True):
        if data.get("kind", "node") != "node":
            continue
        nodes.add(n)
        node_labels[n] = str(data.get("label", n))
        for m in data.get("emits") or []:
            producers[m].add(n)
        consumers[n] = n

    if history:
        for ev in history:
            node = ev.get("node")
            for m in ev.get("emitted") or []:
                if node:
                    producers[m].add(node)

    edge_pairs: Set[Tuple[str, str]] = set()
    for msg, srcs in producers.items():
        v = consumers.get(msg)
        if not v:
            continue
        for u in srcs:
            if u != v:
                edge_pairs.add((u, v))

    return nodes, edge_pairs, node_labels


def _graphviz_pos(H: nx.DiGraph) -> Optional[Dict[Any, Tuple[float, float]]]:
    try:
        from networkx.drawing.nx_agraph import graphviz_layout

        return graphviz_layout(H, prog="dot")
    except Exception:
        pass
    try:
        from networkx.drawing.nx_pydot import graphviz_layout as pydot_layout

        return pydot_layout(H, prog="dot")
    except Exception:
        return None


def _fallback_pos(H: nx.DiGraph) -> Dict[Any, Tuple[float, float]]:
    try:
        order = list(nx.topological_sort(H))
        cols: DefaultDict[int, List[Any]] = defaultdict(list)
        dist = {n: 0 for n in H.nodes()}
        for n in order:
            for _, v in H.out_edges(n):
                dist[v] = max(dist[v], dist[n] + 1)
        for n, d in dist.items():
            cols[d].append(n)
        x_step, y_step = 4.0, 1.6
        pos: Dict[Any, Tuple[float, float]] = {}
        for x, col in enumerate(sorted(cols.keys())):
            for i, n in enumerate(sorted(cols[col], key=str)):
                pos[n] = (x * x_step, -i * y_step)
        return pos
    except Exception:
        pass

    return nx.kamada_kawai_layout(H, weight=None)


def _compute_layout(H: nx.DiGraph) -> Dict[Any, Tuple[float, float]]:
    return _graphviz_pos(H) or _fallback_pos(H)


def render_workflow_graph(
    G: nx.DiGraph,
    *,
    history: Optional[List[Dict[str, Any]]] = None,
    figsize: Tuple[int, int] = (11, 6),
    filename: Optional[str] = None,
    title: Optional[str] = None,
    observed_only: bool = False,
):
    nodes, declared_pairs, node_labels = _declared_edges(G, history)
    observed_pairs = _observed_node_edges(history)

    H = nx.DiGraph()
    for n in nodes:
        H.add_node(n, label=node_labels.get(n, n))

    for u, v in declared_pairs:
        if u in H and v in H:
            H.add_edge(u, v)
    for u, v in observed_pairs:
        if u in H and v in H:
            H.add_edge(u, v)

    if observed_only:
        keep = set()
        for u, v in observed_pairs:
            keep.add(u)
            keep.add(v)
        H = H.subgraph(sorted(keep, key=str)).copy()

    pos = _compute_layout(H)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_axis_off()

    all_edges = sorted(H.edges(), key=lambda e: (str(e[0]), str(e[1])))
    observed_set = set(e for e in all_edges if e in observed_pairs)
    bg_edges = [e for e in all_edges if e not in observed_set]
    hi_edges = [e for e in all_edges if e in observed_set]

    if bg_edges:
        nx.draw_networkx_edges(
            H,
            pos,
            edgelist=bg_edges,
            width=1.4,
            edge_color="#9aa0a6",
            arrows=True,
            arrowsize=16,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.07",
            ax=ax,
        )
    if hi_edges:
        nx.draw_networkx_edges(
            H,
            pos,
            edgelist=hi_edges,
            width=3.0,
            edge_color="#d62728",
            arrows=True,
            arrowsize=18,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.07",
            ax=ax,
        )

    nx.draw_networkx_nodes(
        H,
        pos,
        nodelist=list(H.nodes()),
        node_size=1400,
        node_shape="s",
        node_color="#d2e3fc",
        edgecolors="#202124",
        linewidths=1.2,
        ax=ax,
    )
    labels = {n: H.nodes[n].get("label", str(n)) for n in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=10, ax=ax)

    legend_elems = [
        Patch(facecolor="#d2e3fc", edgecolor="#202124", label="Node"),
        Line2D([0], [0], color="#9aa0a6", lw=1.4, label="Declared route"),
        Line2D([0], [0], color="#d62728", lw=3.0, label="Observed path"),
    ]
    ax.legend(
        handles=legend_elems,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )

    if title:
        ax.set_title(title)
    if filename:
        fig.savefig(filename, bbox_inches="tight")
    return fig, ax


def mermaid_from_graph(
    G: nx.DiGraph,
    *,
    history: Optional[List[Dict[str, Any]]] = None,
    observed_only: bool = False,
) -> str:
    nodes, declared_pairs, node_labels = _declared_edges(G, history)
    observed_pairs = _observed_node_edges(history)

    if observed_only:
        declared_pairs = set()

    lines = ["flowchart LR"]
    for n in sorted(nodes):
        label = node_labels.get(n, n)
        lines.append(f"  {n}[{label}]")

    all_pairs = sorted((declared_pairs | observed_pairs), key=lambda e: (str(e[0]), str(e[1])))
    for u, v in all_pairs:
        if (u, v) in observed_pairs:
            lines.append(f"  {u} ==> {v}")
        else:
            lines.append(f"  {u} --> {v}")

    return "\n".join(lines)
