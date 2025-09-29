import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba_array

from archetypes.visualization.utils import get_cmap

from .utils import map_colors


def stacked_bar(
    points,
    ax=None,
    **params,
):
    """
    A stacked bar plot of *points* with multiple optional parameters to obtain a customized
    visualization.

    Parameters
    ----------
    points : numpy.ndarray
        The points to plot.
    ax : matplotlib.pyplot.axes or None
        The axes to plot on. If None, the current axes will be used.
    **params : dict, optional
        The parameters to pass to the bar plot. ax.bar(..., **params)

    Returns
    -------
    matplotlib.pyplot.axes
        The axes with the plot.
    """
    if not ax:
        ax = plt.gca()

    points = np.asarray(points)
    m = points.shape[0]
    n = points.shape[1]

    labels = np.arange(m)

    params_default = {
        "width": 1,
    }

    if params is None:
        params = params_default

    for k, v in params_default.items():
        params.setdefault(k, v)

    # check if cmap is in params, otherwise use default
    cmap = get_cmap()

    if "color" in params:
        if isinstance(params["color"], str):
            params["color"] = [params["color"]] * n
        elif isinstance(params["color"], list) and len(params["color"]) != n:
            raise ValueError(f"Length of color list must be {n}.")
        color = params.pop("color")
    else:
        color = list(range(n))

    color = map_colors(color, cmap=cmap)

    color[:, -1] = 0.5  # set alpha to 0.5 for bars

    if "edgecolor" in params:
        if isinstance(params["edgecolor"], str):
            params["edgecolor"] = [params["edgecolor"]] * n
        elif isinstance(params["edgecolor"], list) and len(params["edgecolor"]) != n:
            raise ValueError(f"Length of edgecolor list must be {n}.")
        edgecolor = params.pop("edgecolor")
        edgecolor = to_rgba_array(edgecolor)
    else:
        edgecolor = color.copy()
        edgecolor[:, -1] = 1  # set alpha to 1 for edges

    # Plot the points
    bottom = np.zeros(m)
    for j in range(n):
        _ = ax.bar(
            labels,
            points[:, j],
            bottom=bottom,
            color=color[j],
            edgecolor=edgecolor[j],
            label="Archetypes" if j == 0 else None,
            **params,
        )

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
    # Show percentage on y-axis
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_xticklabels([])

    # set axis limits
    ax.axis("on")
    ax.set_aspect("auto")

    return ax
