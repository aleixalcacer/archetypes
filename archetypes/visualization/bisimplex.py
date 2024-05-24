import matplotlib.pyplot as plt
import numpy as np

from .simplex import simplex
from .utils import create_palette


def bisimplex(alphas, archetypes, ax=None, **kwargs):
    """
    A plot of *archetypes* with its corresponding *alphas* in simplex coordinates.

    Parameters
    ----------
    alphas: list of numpy.ndarray
        The archetypal representation of the dataset for each dimension.
    archetypes: numpy.ndarray
        The archetypes.
    ax: matplotlib.pyplot.axes or None
        The axes to plot on. If None, a new figure and axes is created.
    kwargs: dict
        Additional keyword arguments to pass to the simplex plots.

    Returns
    -------
    ax: matplotlib.pyplot.axes
        The axes.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    n_archetypes = archetypes.shape

    # Get the colors for the vertices of the polytopes
    palette = create_palette(
        saturation=0.35, value=0.9, n_colors=n_archetypes[0] + n_archetypes[1], int_colors=1
    )

    colors = palette(np.arange(n_archetypes[0] + n_archetypes[1]))
    colors_1 = colors[: n_archetypes[0]]
    colors_2 = colors[n_archetypes[0] : n_archetypes[0] + n_archetypes[1]]

    # Draw the polytopes
    simplex_kwargs = {
        "show_circle": False,
        "alpha": 0.1,
        "c": "k",
        "linewidth": 0,
        "return_vertices": True,
        "ax": ax,
    }

    alphas1 = alphas[0]
    if len(alphas1) > 1000:
        # pick random 1000 points
        idx = np.random.choice(len(alphas1), 1000, replace=False)
        alphas1 = alphas1[idx]
    alphas2 = alphas[1]
    if len(alphas2) > 1000:
        # pick random 1000 points
        idx = np.random.choice(len(alphas2), 1000, replace=False)
        alphas2 = alphas2[idx]

    _, v1 = simplex(alphas1, origin=(0, -3), vertices_color=colors_1, **simplex_kwargs, **kwargs)
    _, v2 = simplex(alphas2, origin=(3, 0), vertices_color=colors_2, **simplex_kwargs, **kwargs)

    x1, y1 = v1.T
    x2, y2 = v2.T

    # Scale archetypes between 0 and 1
    archetypes_scaled = (archetypes - archetypes.min()) / (archetypes.max() - archetypes.min())

    # Compute the cell size
    sq_size = min(1 / (n_archetypes[0]), 1 / (n_archetypes[1]))
    x_size = sq_size * n_archetypes[1]
    y_size = sq_size * n_archetypes[0]

    # Compute the coordinates of the cells
    x_matrix = np.linspace(-x_size / 2, x_size / 2, n_archetypes[1], endpoint=False)
    y_matrix = np.linspace(-y_size / 2, y_size / 2, n_archetypes[0], endpoint=False)[::-1]

    xx, yy = np.meshgrid(x_matrix[:], y_matrix[:])

    # Draw the archetypes next to the heatmap
    yy, xx = np.meshgrid(x_matrix[0], y_matrix)
    for y_i, x_i, c_i in zip(xx.flatten(), yy.flatten(), colors_1):
        square = plt.Rectangle(
            (x_i - 3 / 2 * sq_size, y_i - sq_size / 2),
            sq_size / 4,
            sq_size,
            color=c_i,
            fill=True,
            linewidth=0,
        )
        ax.add_patch(square)

    yy, xx = np.meshgrid(x_matrix, y_matrix[0])
    for (y_i, x_i), c_i in zip(zip(xx.flatten(), yy.flatten()), colors_2):
        square = plt.Rectangle(
            (x_i - sq_size / 2, y_i + 5 / 4 * sq_size),
            sq_size,
            sq_size / 4,
            color=c_i,
            fill=True,
            linewidth=0,
        )
        ax.add_patch(square)

    # Draw colorbar as rectangle
    ax2 = ax.inset_axes(
        [-x_size / 2 - sq_size / 2, -y_size / 2 - sq_size / 2, x_size, y_size],
        transform=ax.transData,
        zorder=-10,
    )

    # heatmap(archetypes_scaled, ax=ax2, **kwargs)

    ax2.imshow(archetypes_scaled, cmap=plt.cm.Greys, vmin=0, vmax=1)
    ax2.axis("off")

    # Draw lines between archetypes and heatmap cells
    for i, (x_i, y_i, c1_i) in enumerate(zip(x1, y1, colors_1)):
        for j, (x_matrix_i, c2_i) in enumerate(zip(x_matrix, colors_2)):
            ax.plot(
                [x_i, x_matrix_i],
                [y_i, y_matrix[i]],
                color=c1_i,
                linestyle="-",
                linewidth=archetypes_scaled[i, j] * 5,
                zorder=0,
                alpha=archetypes_scaled[i, j] ** 2,
            )

    for i, (x_i, y_i, c2_i) in enumerate(zip(x2, y2, colors_2)):
        for j, (y_matrix_i, c1_i) in enumerate(zip(y_matrix, colors_1)):
            ax.plot(
                [x_i, x_matrix[i]],
                [y_i, y_matrix_i],
                color=c2_i,
                linestyle="-",
                linewidth=archetypes_scaled[j, i] * 5,
                zorder=0,
                alpha=archetypes_scaled[j, i] ** 2,
            )

    # set aspect ratio to equal
    ax.set_aspect("equal")

    # set axis off
    ax.axis("off")

    # set axis limits
    ax.autoscale(enable=True, axis="both", tight=False)

    ax.figure.tight_layout()

    return ax
