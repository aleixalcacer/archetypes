import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


def simplex(
    points,
    origin=(0, 0),
    show_points=True,
    show_direction=False,
    direction_color="black",
    direction_alpha=0.2,
    show_edges=True,
    show_circle=False,
    ax=None,
    labels=None,
    show_vertices=False,
    color="lightgray",
    vertices_color=None,
    vertices_size=None,
    vertices_labels=None,
    return_vertices=False,
    **kwargs,
):
    """
    A simplex plot of *points* with multiple optional parameters to obtain a customized
    visualization.

    Parameters
    ----------
    points : numpy.ndarray
        The points to plot.
    origin : tuple of float
        The origin of the simplex in the plot.
    show_points : bool
        If True, *points* are displayed.
    show_direction : bool
        If True, direction arrows are displayed.
    direction_color : color
        Set the color of the direction arrows.
    direction_alpha: scalar or None
        Set the alpha of the direction arrows. It must be within the 0-1 range, inclusive.
    show_edges : bool
        If True, the edges of the plot are displayed.
    show_circle : bool
        If True, the circle of the plot are displayed.
    ax : matplotlib.pyplot.axes or None
        The axes
    labels : list or None
        A list of labels for the points. If None, no labels are displayed.
    show_vertices : bool
        If True, the vertices of the simplex are displayed.
    vertices_color : color
        Set the color of the vertices.
    vertices_size : scalar
        Set the size of the vertices.
    vertices_labels : list or None
        A list of labels for the vertices. If None, no labels are displayed.
    return_vertices : bool
        If True, the coordinates of the vertices are returned along with the axes.
        If False, only the axes are returned.
    kwargs

    """
    if not ax:
        ax = plt.gca()

    points = np.asarray(points)

    n = points.shape[1]

    # Set the background
    from itertools import combinations

    from matplotlib.patches import Ellipse, PathPatch

    if show_circle:
        circle = Ellipse(
            origin, 2, 2, linewidth=1, edgecolor="lightgray", facecolor="none", zorder=0
        )
        ax.add_patch(circle)

    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(theta) + origin[0]
    y = np.sin(theta) + origin[1]

    vertices = np.stack((x, y), axis=1)

    if show_edges:
        edges = combinations(vertices, 2)
        for p1, p2 in edges:
            x1, y1 = p1
            x2, y2 = p2
            ax.plot([x1, x2], [y1, y2], "-", linewidth=1, color="lightgray", zorder=0)

    # ax.plot(vertices[:, 0], vertices[:, 1], "o", color="black", alpha=1)
    # Plot vertices

    if vertices_color is None:
        vertices_color = [f"C{i}" for i in range(n)]
    if isinstance(vertices_color, str):
        vertices_color = [vertices_color] * n

    if vertices_size is None:
        vertices_size = rcParams["lines.markersize"] ** 2
    if isinstance(vertices_size, (int, float)):
        vertices_size = [vertices_size] * n

    if vertices_labels is None:
        vertices_labels = [f"{i}" for i in range(n)]

    if show_vertices:
        for i, p in enumerate(vertices[:]):
            ax.scatter(
                p[0],
                p[1],
                zorder=3,
                s=vertices_size[i],
                c=vertices_color[i],
                label=vertices_labels[i],
            )

    # Project the points to 2D
    points_projected = np.apply_along_axis(
        lambda x: np.sum(x.reshape(-1, 1) * vertices, 0), 1, points
    )

    if show_points:
        ax.scatter(points_projected[:, 0], points_projected[:, 1], zorder=2, color=color, **kwargs)

    if labels is not None:
        for i, p in enumerate(points_projected):
            ann_i = ax.annotate(
                labels[i],
                xy=p,
                xytext=(p[0] + 0.03, p[1] + 0.03),
                horizontalalignment="center",
                verticalalignment="center",
                color="gray",
                zorder=3,
            )
            bbox = ann_i.get_window_extent()
            bbox_data = bbox.transformed(ax.transData.inverted())
            ax.update_datalim(bbox_data.corners())

    # Draw the projections to each axis
    if show_direction:
        from matplotlib.path import Path

        for p, p_projected in zip(points, points_projected):
            projections = np.apply_along_axis(lambda x: x - p_projected, 1, vertices)
            projections /= np.linalg.norm(projections, axis=1)[:, None]
            projections *= p[:, None]
            projections = projections / 5
            zeros = np.zeros_like(projections)

            verts = np.stack((zeros, projections), axis=1).reshape(-1, projections.shape[1])
            verts = np.apply_along_axis(lambda x: x + p_projected, 1, verts)

            codes = np.array([Path.MOVETO, Path.LINETO] * n)

            path = Path(verts, codes)
            patch = PathPatch(
                path, edgecolor=direction_color, lw=1, alpha=direction_alpha, zorder=3
            )
            ax.add_patch(patch)

    if show_vertices:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            title="Archetypes",
            frameon=False,
            handlelength=1,
            handleheight=1,
        )

    ax.axis("off")
    aspect = (10 * 0.8) / (8 * 0.9)

    # get ax size

    we = ax.get_window_extent()

    aspect = we.width / we.height

    ax.set_aspect(1)
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(y_min * aspect, y_max * aspect)

    if return_vertices:
        return ax, vertices

    return ax
