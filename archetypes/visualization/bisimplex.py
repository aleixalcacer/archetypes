import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from archetypes.visualization import simplex


def _create_palette(saturation, value, n_colors, int_colors=3):
    hue = np.linspace(0, 1, n_colors, endpoint=False)
    hue = np.hstack([hue[i::int_colors] for i in range(int_colors)])
    saturation = np.full(n_colors, saturation)
    value = np.full(n_colors, value)
    # convert to RGB
    c = mpl.colors.hsv_to_rgb(np.vstack([hue, saturation, value]).T)
    # Create palette
    palette = mpl.colors.ListedColormap(c)
    return palette


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
    palette = _create_palette(
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
    _, v1 = simplex(alphas[0], origin=(0, -3), vertices_color=colors_1, **simplex_kwargs, **kwargs)
    _, v2 = simplex(alphas[1], origin=(3, 0), vertices_color=colors_2, **simplex_kwargs, **kwargs)

    x1, y1 = v1.T
    x2, y2 = v2.T

    # Scale archetypes between 0 and 1
    archetypes_scaled = (archetypes - archetypes.min()) / (archetypes.max() - archetypes.min())

    # Compute the cell size
    sq_size = min(1 / (n_archetypes[0]), 1 / (n_archetypes[1]))
    x_size = sq_size * n_archetypes[1]
    y_size = sq_size * n_archetypes[0]

    # Compute the coordinates of the cells
    x_matrix = np.linspace(-x_size / 2, x_size / 2, n_archetypes[1])
    y_matrix = np.linspace(-y_size / 2, y_size / 2, n_archetypes[0])[::-1]

    xx, yy = np.meshgrid(x_matrix[:], y_matrix[:])

    # Draw the heatmap cells

    # Use grayscale palette from matplotlib between 0 and 1
    # archetypes[:] = .6
    palette = mpl.colormaps["Grays"]

    for x_i, y_i, c_i, a_i in zip(
        xx.flatten(), yy.flatten(), archetypes.flatten(), archetypes_scaled.flatten()
    ):
        # transform black color with alpha to rgba
        c_i = palette(c_i)
        sq_size_i = sq_size * a_i
        square = plt.Rectangle(
            (x_i - sq_size_i / 2, y_i - sq_size_i / 2),
            sq_size_i,
            sq_size_i,
            color=c_i,
            fill=True,
            linewidth=0,
            zorder=10,
        )
        ax.add_patch(square)

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

    # Draw lines between archetypes and heatmap cells
    for i, (x_i, y_i, c1_i) in enumerate(zip(x1, y1, colors_1)):
        for j, (x_matrix_i, c2_i) in enumerate(zip(x_matrix, colors_2)):
            ax.plot(
                [x_i, x_matrix_i],
                [y_i, y_matrix[i]],
                color=c1_i,
                linestyle="-",
                linewidth=archetypes_scaled[i, j] * 7,
                zorder=0,
                alpha=archetypes_scaled[i, j] * 0.5,
            )

    for i, (x_i, y_i, c2_i) in enumerate(zip(x2, y2, colors_2)):
        for j, (y_matrix_i, c1_i) in enumerate(zip(y_matrix, colors_1)):
            ax.plot(
                [x_i, x_matrix[i]],
                [y_i, y_matrix_i],
                color=c2_i,
                linestyle="-",
                linewidth=archetypes_scaled[j, i] * 7,
                zorder=0,
                alpha=archetypes_scaled[j, i] * 0.5,
            )

    # set aspect ratio to equal
    ax.set_aspect("equal")

    # set axis off
    ax.axis("off")

    # set axis limits
    ax.autoscale(enable=True, axis="both", tight=False)

    return ax
