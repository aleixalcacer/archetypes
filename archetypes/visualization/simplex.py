from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from archetypes.visualization.utils import get_cmap, map_colors


def simplex(
    points,
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
    A simplex plot of *points* with multiple optional parameters to obtain a customized
    visualization.

    Parameters
    ----------
    points : numpy.ndarray
        The points to plot.
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
        The parameters to pass to the points plot. ax.scatter(..., **params)
    """
    if not ax:
        ax = plt.gca()

    points = np.asarray(points)
    n = points.shape[1]

    # Draw simplex structure

    # Compute the vertices of the simplex
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(theta) + origin[0]
    y = np.sin(theta) + origin[1]

    vertices = np.stack((x, y), axis=1)

    if show_axis:
        axis_kwargs_default = {
            "color": "black",
            "linewidth": 1,
            "linestyle": "-",
            "zorder": 0,
        }

        if axis_params is None:
            axis_params = axis_kwargs_default

        if "c" in axis_params:
            raise ValueError("Use 'color' instead of 'c' in simplex_kwargs.")

        for k, v in axis_kwargs_default.items():
            axis_params.setdefault(k, v)
        edges = combinations(vertices, 2)

        for p1, p2 in edges:
            x1, y1 = p1
            x2, y2 = p2
            ax.plot([x1, x2], [y1, y2], **axis_params)

    # Draw vertices
    cmap = get_cmap()

    if show_vertices:
        vertices_kwargs_default = {
            "zorder": 2,
            "s": getattr(rcParams, "lines.markersize", 50) * 2,
            "label": "Archetypes",
            "c": list(range(n)),
            "cmap": cmap,
        }

        if vertices_params is None:
            vertices_params = vertices_kwargs_default

        if "color" in vertices_params:
            vertices_kwargs_default.pop("c")

        for k, v in vertices_kwargs_default.items():
            vertices_params.setdefault(k, v)

        if "c" in vertices_params:
            vertices_params["c"] = map_colors(
                vertices_params["c"], cmap=vertices_params.get("cmap", cmap)
            )
            vertices_params.pop("cmap", None)
        ax.scatter(
            *vertices.T,
            **vertices_params,
        )

    # Project the points to 2D
    points_projected = np.apply_along_axis(
        lambda x: np.sum(x.reshape(-1, 1) * vertices, 0), 1, points
    )

    kwargs_default = {
        "c": "lightgray",
        "label": "Observations",
    }

    if "color" in params:
        kwargs_default.pop("c")

    for k, v in kwargs_default.items():
        params.setdefault(k, v)

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
