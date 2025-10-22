import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from archetypes.processing import get_closest_n, sort_by_coefficients


def heatmap(data, coefficients=None, n=None, reorder=False, ax=None, **kwargs):
    """
    Plot a heatmap of the data.

    If coefficients are provided, the heatmap is reordered according to the coefficients.

    Parameters
    ----------
    data: np.ndarray
        The data to plot.
    coefficients: list of np.ndarray or None, default=None
        The coefficients to use for sorting the data. If None, no sorting is applied.
    n: int or None, default=None
        If provided, only the n closest samples to each archetype are shown. Requires
        coefficients to be provided. If None, all samples are shown.
    reorder: bool, optional
        Whether to reorder the archetypal groups by size.
        Default is False.
    ax: matplotlib.pyplot.axes or None
        The axes to plot on. If None, a new figure and axes is created.
    **params : dict, optional
        The parameters to pass to the matshow plot. ax.matshow(..., **params)
    Returns
    -------
    ax: matplotlib.pyplot.axes
        The axes.
    """
    if coefficients:
        data, coefficients, perms = sort_by_coefficients(data, coefficients, reorder=reorder)
        if n:
            # Check n is a positive integer
            if not isinstance(n, int):
                raise ValueError(f"n must be an integer, got {type(n)}")
            if n <= 0:
                raise ValueError(f"n must be a positive integer, got {n}")

            data, coefficients, perms_closest = get_closest_n(data, coefficients, n=n)
            perms = [p[p_c] for p, p_c in zip(perms, perms_closest)]

        labels = [np.argmax(c, axis=1) for c in coefficients]

    else:
        perms = [np.arange(data.shape[i]) for i in range(data.ndim)]
        if n is not None:
            raise ValueError("nclosest requires coefficients to be provided")
        labels = None

    data = np.asarray(data)

    # check data is 2D
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got {data.ndim}D")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    if "cmap" not in kwargs:
        kwargs["cmap"] = "Greys"
    if "aspect" not in kwargs:
        kwargs["aspect"] = "auto"

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

    ax.matshow(data, interpolation="none", **kwargs)

    # ax.set_aspect("auto")

    ax.set_xticks([])
    ax.set_yticks([])

    return ax
