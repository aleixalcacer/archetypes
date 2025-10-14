from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from archetypes.visualization.utils import get_cmap, process_params


def simplex(
    data,
    show_axis=True,
    axis_params=None,
    show_vertices=True,
    vertices_params=None,
    return_vertices=False,
    origin=(0, 0),
    ax=None,
    **params,
):
    """
    A simplex plot of *data* with multiple optional parameters to obtain a customized
    visualization.

    Parameters
    ----------
    data : numpy.ndarray
        The data to plot.
    show_axis : bool, optional
        Whether to show the simplex axis, by default True.
    axis_params : dict, optional
        The parameters to pass to the axis plot. ax.plot(...,**axis_params)
    show_vertices : bool, optional
        Whether to show the simplex vertices, i.e., archetypes, by default True.
    vertices_params : dict, optional
        The parameters to pass to the vertices plot. ax.scatter(..., **vertices_params)
    return_vertices : bool, optional
        Whether to return the vertices of the simplex, by default False.
    origin : tuple, optional
        The origin of the simplex, by default (0, 0).
    ax : matplotlib.axes.Axes, optional
        The axes to plot on, by default None. If None, the current axes will be used.
    **params : dict, optional
        The parameters to pass to the scatter plot. ax.scatter(..., **params)
    """
    if not ax:
        ax = plt.gca()

    data = np.asarray(data)

    # Ckeck if data is a simplex
    if np.any(data < 0):
        raise ValueError("data must be non-negative")
    if not np.allclose(data.sum(axis=1), 1):
        raise ValueError("rows must sum to 1")

    m = data.shape[1]
    n = data.shape[0]

    cmap = get_cmap()

    # Compute the vertices of the simplex
    theta = np.linspace(0, 2 * np.pi, m, endpoint=False)
    x = np.cos(theta) + origin[0]
    y = np.sin(theta) + origin[1]

    vertices = np.stack((x, y), axis=1)

    # Draw the simplex axis
    if show_axis:
        axis_params_default = {
            "color": "black",
            "linewidth": 1,
            "linestyle": "-",
            "zorder": 0,
            "cmap": cmap,
        }

        axis_params = process_params(1, axis_params, axis_params_default)

        edges = combinations(vertices, 2)

        for p1, p2 in edges:
            x1, y1 = p1
            x2, y2 = p2
            ax.plot([x1, x2], [y1, y2], **axis_params)

    # Draw the simplex vertices
    if show_vertices:
        vertices_params_default = {
            "zorder": 2,
            "s": getattr(rcParams, "lines.markersize", 50) * 2,
            "label": "Archetypes",
            "c": list(range(m)),
            "cmap": cmap,
        }

        vertices_params = process_params(m, vertices_params, vertices_params_default)

        ax.scatter(
            *vertices.T,
            **vertices_params,
        )

    # Project the points to 2D
    points_projected = np.apply_along_axis(
        lambda x: np.sum(x.reshape(-1, 1) * vertices, 0), 1, data
    )

    # Draw the points
    params_default = {
        "c": "lightgray",
        "label": "Observations",
        "cmap": cmap,
    }

    params = process_params(n, params, params_default)

    ax.scatter(points_projected[:, 0], points_projected[:, 1], **params)

    # TODO: Check this transformations
    ax.axis("off")

    we = ax.get_window_extent()
    aspect = we.width / we.height

    ax.set_aspect(1)

    # set limits
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(y_min * aspect, y_max * aspect)

    if return_vertices:
        return ax, vertices

    return ax
