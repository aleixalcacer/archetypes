import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Polygon

from .utils import create_palette


def heatmap(data, labels=None, n_archetypes=None, scores=None, ax=None, **kwargs):
    """Plot a heatmap of the data. If labels are provided, the heatmap is divided into cells.

    Parameters
    ----------
    data: np.ndarray
        The data to plot.
    labels: list of np.ndarray or None
        The labels values to use for the plot.
        If None, the labels values are computed from the labels.
    n_archetypes: list of int or None
        The number of archetypes for each dimension.
        If None, the number of archetypes is computed from the labels.
    scores: list of np.ndarray or None
        The scores values to use for the plot.
    ax: matplotlib.pyplot.axes or None
        The axes to plot on. If None, a new figure and axes is created.
    kwargs: dict
        Additional keyword arguments to pass to the heatmap.

    Returns
    -------
    ax: matplotlib.pyplot.axes
        The axes.
    """

    data = np.asarray(data)
    scores = [np.asarray(s) for s in scores] if scores is not None else None
    # check data is 2D
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got {data.ndim}D")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    if "cmap" not in kwargs:
        kwargs["cmap"] = "Greys"

    # Plot line if labels[i] != labels[i+1]
    if labels is not None:
        # check labels is a list of 2 arrays
        if not isinstance(labels, list):
            raise ValueError(f"labels must be a list of 2 arrays, got {type(labels)}")

        if len(labels) != 2:
            raise ValueError(f"labels must be a list of 2 arrays, got {len(labels)} arrays")

        if not isinstance(labels[0], np.ndarray) or not isinstance(labels[1], np.ndarray):
            raise ValueError(
                f"labels must be a list of 2 arrays, got {type(labels[0])} and {type(labels[1])}"
            )

        labels_h = np.concatenate([labels[1].flatten()])
        labels_v = np.concatenate([labels[0].flatten()])

        polygon_kwargs = {"color": "k", "lw": 1}

        for i in range(len(labels_h) - 1):
            if labels_h[i] != labels_h[i + 1]:
                line = Polygon(
                    np.array([[i + 1, 0], [i + 1, data.shape[0]]]) - 0.5, **polygon_kwargs
                )
                ax.add_patch(line)

        for i in range(len(labels_v) - 1):
            if labels_v[i] != labels_v[i + 1]:
                line = Polygon(
                    np.array([[0, i + 1], [data.shape[1], i + 1]]) - 0.5, **polygon_kwargs
                )
                ax.add_patch(line)

        if n_archetypes is None:
            n_archetypes = [len(np.unique(labels[0])), len(np.unique(labels[1]))]

        # Add a rectangle to frame the data
        rect = Polygon(
            np.array(
                [[0, 0], [data.shape[1], 0], [data.shape[1], data.shape[0]], [0, data.shape[0]]]
            )
            - 0.5,
            fill=False,
            **polygon_kwargs,
        )
        ax.add_patch(rect)

        palette = create_palette(
            saturation=0.35, value=0.9, n_colors=n_archetypes[0] + n_archetypes[1], int_colors=1
        )
        colors = palette(np.arange(n_archetypes[0] + n_archetypes[1]))
        colors_1 = colors[: n_archetypes[0]]
        colors_2 = colors[n_archetypes[0] : n_archetypes[0] + n_archetypes[1]]

        # Plot archetypes
        counts = [np.count_nonzero(labels[0] == i) for i in range(n_archetypes[0])]
        counts = np.cumsum(counts)
        counts = np.concatenate([[0], counts]) - 0.5

        arch_factor = 0.05 * data.shape[1]

        if scores is None:
            scores = [np.ones_like(labels[0]), np.ones_like(labels[1])]

        for c, (i0, i1) in zip(colors_1, zip(counts, counts[1:])):
            c1 = np.array(to_rgb(c))
            c2 = np.array([1, 1, 1])

            ax.imshow(
                scores[0][int(i0 + 0.5) : int(i1 + 0.5)][::-1].reshape(-1, 1),
                extent=[-0.5 - arch_factor, -0.5 - 2 * arch_factor, i0, i1],
                cmap=LinearSegmentedColormap.from_list("c", [c2, c1]),
                interpolation="none",
                vmax=1,
                vmin=0,
            )

        counts = [np.count_nonzero(labels[1] == i) for i in range(n_archetypes[1])]
        counts = np.cumsum(counts)
        counts = np.concatenate([[0], counts]) - 0.5

        arch_factor = 0.05 * data.shape[0]

        for c, (i0, i1) in zip(colors_2, zip(counts, counts[1:])):
            c1 = np.array(to_rgb(c))
            c2 = np.array([1, 1, 1])

            ax.imshow(
                scores[1][int(i0 + 0.5) : int(i1 + 0.5)].reshape(1, -1),
                extent=[i0, i1, -0.5 - arch_factor, -0.5 - 2 * arch_factor],
                cmap=LinearSegmentedColormap.from_list("c", [c2, c1]),
                interpolation="none",
                vmax=1,
                vmin=0,
            )

    ax.matshow(data, interpolation="none", **kwargs)

    # set aspect ratio to square
    ax.set_box_aspect(1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    ax.set_xlim(-2 * data.shape[1] * 0.05 - 0.5, data.shape[1] - 0.5)
    ax.set_ylim(data.shape[0] - 0.5, -2 * data.shape[0] * 0.05 - 0.5)
    ax.set_aspect("auto")

    # set axis off
    ax.axis("off")
    # set axis limits

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    #
    lim_factor = 0.01
    ax.set_xlim(xlim[0] - lim_factor * data.shape[1], xlim[1] + lim_factor * data.shape[1])
    ax.set_ylim(ylim[0] + lim_factor * data.shape[0], ylim[1] - lim_factor * data.shape[0])

    return ax
