import matplotlib.pyplot as plt
import numpy as np


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
    vertices_color="k",
    vertices_size=200,
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
    if show_vertices:
        ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            s=vertices_size,
            c=vertices_color,
            zorder=1,
        )

    if vertices_labels is None:
        vertices_labels = [f"A{i}" for i in range(n)]

    ax.set_xlim(-2 + origin[0], 2 + origin[0])
    ax.set_ylim(-2 + origin[1], 2 + origin[1])

    annotations = []
    for i, p in enumerate(vertices[:]):
        ann_i = ax.annotate(
            vertices_labels[i],
            xy=(p - origin) * 1.1 + origin,
            xytext=(p - origin) * 1.3 + origin,
            arrowprops={
                "arrowstyle": "->",
                "lw": 1,
                "color": "gray",
                "connectionstyle": "arc3,rad=0.2",
            },
            horizontalalignment="center",
            verticalalignment="center",
            zorder=3,
            transform=ax.transData,
            # bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1),
        )
        annotations.append(ann_i)

    # get renderer
    renderer = ax.figure.canvas.get_renderer()

    min_y, max_y = np.inf, -np.inf
    min_x, max_x = np.inf, -np.inf
    for ann_i in annotations:
        bbox = ann_i.get_window_extent(renderer=renderer)
        bbox_data = bbox.transformed(ax.transData.inverted())
        corners = bbox_data.corners()
        min_y = min(min_y, corners[:, 1].min())
        max_y = max(max_y, corners[:, 1].max())
        min_x = min(min_x, corners[:, 0].min())
        max_x = max(max_x, corners[:, 0].max())

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Project the points to 2D
    points_projected = np.apply_along_axis(
        lambda x: np.sum(x.reshape(-1, 1) * vertices, 0), 1, points
    )

    if show_points:
        ax.scatter(points_projected[:, 0], points_projected[:, 1], zorder=2, **kwargs)

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

    ax.axis("off")
    ax.set_aspect("equal")
    ax.autoscale()
    # get datalim

    bbox = ax.get_window_extent(renderer=renderer)
    bbox_data = bbox.transformed(ax.transData.inverted())
    corners = bbox_data.corners()

    # plot bbox
    ax.scatter(corners[:, 0], corners[:, 1], marker="")

    if return_vertices:
        return ax, vertices

    return ax
