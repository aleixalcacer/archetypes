import matplotlib.pyplot as plt
import numpy as np


def stacked_bar(
    points,
    ax=None,
    labels=None,
    vertices_labels=None,
    **kwargs,
):
    """
    A stacked bar plot of *points* with multiple optional parameters to obtain a customized
    visualization.

    Parameters
    ----------
    points : numpy.ndarray
        The points to plot.
    ax : matplotlib.pyplot.axes or None
        The axes
    labels : list or None
        A list of labels for the points. If None, no labels are displayed.
    vertices_labels : list or None
        A list of labels for the vertices. If None, no labels are displayed.
    kwargs

    """
    if not ax:
        ax = plt.gca()

    points = np.asarray(points)
    m = points.shape[0]
    n = points.shape[1]

    if vertices_labels is None:
        vertices_labels = [f"{i}" for i in range(n)]

    if labels is None:
        show_labels = False
        labels = np.arange(m)
    else:
        show_labels = True

    # Plot the points
    bottom = np.zeros(m)
    for j, vertice_label in enumerate(vertices_labels):
        ax.bar(labels, points[:, j], label=vertice_label, bottom=bottom, **kwargs)
        bottom += points[:, j]

    # remove background
    ax.set_facecolor("none")

    # Set the labels
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.set_ylabel("Similarity degree")
    ax.set_xlabel("Observations")
    ax.set_yticks([])
    ax.set_yticklabels([])
    if not show_labels:
        ax.set_xticks([])
        ax.set_xticklabels([])

    # set axis limits
    ax.axis("on")

    ax.set_aspect("auto")
    ax.autoscale()

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Archetypes",
        frameon=False,
        handlelength=1,
        handleheight=1,
    )

    return ax
