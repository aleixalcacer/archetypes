import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def heatmap(data, labels=None, ax=None, **kwargs):
    """Plot a heatmap of the data. If labels are provided, the heatmap is divided into cells.

    Parameters
    ----------
    data: np.ndarray
        The data to plot.
    labels: list of np.ndarray or None
        The labels to use to divide the heatmap into cells. If None, no labels are used.
    ax: matplotlib.pyplot.axes or None
        The axes to plot on. If None, a new figure and axes is created.
    kwargs: dict
        Additional keyword arguments to pass to the heatmap.

    Returns
    -------
    ax: matplotlib.pyplot.axes
        The axes.
    """

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

        labels_h = np.concatenate([[-1], labels[1].flatten(), [-1]])
        labels_v = np.concatenate([[-1], labels[0].flatten(), [-1]])

        polygon_kwargs = {"color": "r", "lw": 1}

        for i in range(len(labels_h) - 1):
            if labels_h[i] != labels_h[i + 1]:
                line = Polygon(np.array([[i, 0], [i, data.shape[0]]]) - 0.5, **polygon_kwargs)
                ax.add_patch(line)

        for i in range(len(labels_v) - 1):
            if labels_v[i] != labels_v[i + 1]:
                line = Polygon(np.array([[0, i], [data.shape[1], i]]) - 0.5, **polygon_kwargs)
                ax.add_patch(line)

    ax.matshow(data, rasterized=True, **kwargs)

    # set aspect ratio to equal
    ax.set_aspect("equal")
    # set axis off
    ax.axis("off")
    # set axis limits

    ax.autoscale(enable=True, axis="both", tight=False)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    exp_factor = 0.01 * max(np.abs(np.diff(xlim)), np.abs(np.diff(ylim)))

    ax.set_xlim(xlim[0] - exp_factor, xlim[1] + exp_factor)
    ax.set_ylim(ylim[0] + exp_factor, ylim[1] - exp_factor)

    return ax
