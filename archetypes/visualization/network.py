import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def network(
    points,
    threshold=0.3,
    ax=None,
    labels=None,
    color="lightgray",
    vertices_labels=None,
    **kwargs,
):
    """
    A network of *points* with multiple optional parameters to obtain a customized
    visualization.

    Parameters
    ----------
    points : numpy.ndarray
        The points to plot.
    ax : matplotlib.pyplot.axes or None
        The axes
    labels : list or None
        A list of labels for the points. If None, no labels are displayed.
    color: str or list
        The color of the points. If a list, it must have the same length as the number of points.
    vertices_labels : list or None
        A list of labels for the vertices. If None, no labels are displayed.
    kwargs

    """
    if not ax:
        ax = plt.gca()

    points = np.asarray(points)

    m = points.shape[0]
    n = points.shape[1]

    if isinstance(color, str):
        color = [color] * m
    else:
        assert len(color) == m, "The color list must have the same length as the number of points"

    if vertices_labels is None:
        vertices_labels = [f"{i}" for i in range(n)]

    if labels is None:
        show_labels = False
        labels = np.arange(m)
    else:
        show_labels = True

    # Create the graph

    G = nx.Graph()
    for i in range(m):
        G.add_node(i, label=labels[i], color=color[i])

    for i in range(n):
        G.add_node(i + m, label=vertices_labels[i], color=f"C{i}")

    # Add all edges
    for i in range(m):
        for j in range(n):
            if points[i, j] > threshold:
                G.add_edge(i, j + m, weight=points[i, j])

    # drop nodes with no edges
    nodes_to_drop = [node for node in G.nodes if G.degree(node) == 0]
    G.remove_nodes_from(nodes_to_drop)

    with warnings.catch_warnings(action="ignore"):
        pos = nx.nx_agraph.graphviz_layout(G)

    nodes = G.nodes()

    points = list(nodes)[:-n]

    color_map = [G.nodes[n]["color"] for n in points]
    labels_map = {n: G.nodes[n]["label"] for n in points}
    weight_map = [G.edges[(u, v)]["weight"] for u, v in G.edges]

    nx.draw(
        G,
        pos,
        nodelist=points,
        ax=ax,
        node_color=color_map,
        width=weight_map,
        edge_color="gray",
        labels=labels_map,
        node_size=100,
        with_labels=show_labels,
        **kwargs,
    )

    archetypes = list(nodes)[-n:]

    color_map = [G.nodes[n]["color"] for n in archetypes]
    labels_map = {n: G.nodes[n]["label"] for n in archetypes}

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=archetypes,
        ax=ax,
        node_color=color_map,
        node_size=300,
        alpha=1.0,
    )

    # add legend
    for i in range(n):
        ax.scatter([], [], color=f"C{i}", s=100, label=i)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Archetypes",
        frameon=False,
        handlelength=1,
        handleheight=1,
    )
    # ax.set_aspect(1)
    ax.autoscale()

    return ax
