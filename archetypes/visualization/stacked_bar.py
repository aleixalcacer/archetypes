import matplotlib.pyplot as plt
import numpy as np

from .utils import get_cmap, process_params


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

    cmap = get_cmap()

    params_default = {
        "width": 1,
        "cmap": cmap,
        "c": range(n),
        "label": "Archetypes",
    }

    if "edgecolor" not in params:
        if "color" in params:
            params["edgecolor"] = params["color"].copy()
        elif "c" in params:
            params["edgecolor"] = params["c"].copy()
        else:
            params_default["edgecolor"] = range(n)

    # check if cmap is in params, otherwise use default
    params = process_params(n, params, params_default)
    edgecolor = params.pop("edgecolor")
    if "color" in params:
        color = params.pop("color")
    else:
        color = params.pop("c")
    color[:, -1] = 0.5  # set alpha to 0.5

    label = params.pop("label")

    # Plot the points
    bottom = np.zeros(m)
    for j in range(n):
        _ = ax.bar(
            labels,
            points[:, j],
            bottom=bottom,
            color=color[j],
            edgecolor=edgecolor[j],
            label=label if j == 0 else None,
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

    ax.set_ylabel("Mixture Coefficients")
    ax.set_xlabel("Samples")
    ax.set_yticks([])
    # Show percentage on y-axis
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_xticklabels([])

    # set axis limits
    ax.axis("on")
    ax.set_aspect("auto")

    return ax
