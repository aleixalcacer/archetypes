import matplotlib.pyplot as plt
import numpy as np


def circular_bar(
    archetypes,
    data=None,
    ax=None,
    show_labels=True,
    labels=None,
    vertices_labels=None,
    **kwargs,
):
    """
    A stacked bar plot of *points* with multiple optional parameters to obtain a customized
    visualization.

    Parameters
    ----------
    archetypes : numpy.ndarray
        The archetypes to plot.
    data : numpy.ndarray or None
        If not None, the data to standardize the archetypes.
    kind : str
        The kind of plot. It can be "bar" or "line".
    ax : List[matplotlib.axes.Axes] or None
        A list of matplotlib axes to plot the archetypes. If None, a new figure is created.
    labels : list or None
        A list of labels for the variables. If None, no labels are displayed.
    vertices_labels : list or None
        A list of labels for the vertices. If None, no labels are displayed.
    kwargs
    """
    archetypes = np.asarray(archetypes)
    m = archetypes.shape[0]
    n = archetypes.shape[1]

    if ax is None:
        fig = plt.gcf()
        ax = [fig.add_subplot(1, m, i + 1) for i in range(m)]

    # Normalize the archetypes if data is provided
    data = np.asarray(data)
    if data is not None:
        archetypes = (data[np.newaxis, :, :] < archetypes[:, np.newaxis, :]).mean(axis=1)

    # Check archetypes are in [0, 1]
    assert np.all((archetypes >= 0) & (archetypes <= 1)), "Archetypes must be in [0, 1]"

    if labels is None:
        labels = np.arange(n)
    # color = [f"C{i}" for i in range(n)]

    if vertices_labels is None:
        vertices_labels = [f"{i}" for i in range(m)]

    lower_bound = 10
    max_bound = 50
    slope = max_bound - lower_bound
    label_padding = lower_bound + 4

    width = 2 * np.pi / n

    max_y = 0
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

    axs_legend = []
    axs_polar = []
    for i in range(m):
        # Convert ax to polar projection
        cartesian_ax = ax[i]
        subplot_spec = cartesian_ax.get_subplotspec()
        polar_ax = plt.subplot(subplot_spec, polar=True)
        polar_ax.set_ylim(0, max_bound + 10)

        axs_legend.append(cartesian_ax)
        axs_polar.append(polar_ax)

        heights = lower_bound + slope * archetypes[i]

        _ = polar_ax.bar(
            angles,
            heights,
            width=width,
            bottom=lower_bound,
            color="whitesmoke",
            edgecolor="lightgray",
            **kwargs,
        )
        _ = polar_ax.bar([0], lower_bound, width=2 * np.pi, color=f"C{i}")

        # This is the space between the end of the bar and the label
        if show_labels:
            # Iterate over angles, values, and labels, to add all of them.
            for angle, height, label in zip(angles, heights, labels):

                # Labels are rotated. Rotation must be specified in degrees :(
                rotation = np.rad2deg(angle)

                # Flip some labels upside down
                alignment = ""
                if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
                    alignment = "right"
                    rotation = rotation + 180
                else:
                    alignment = "left"

                # And finally add the text
                ann_i = polar_ax.text(
                    x=angle,
                    y=label_padding,
                    fontdict={"fontsize": 8},
                    s=label,
                    ha=alignment,
                    va="center",
                    rotation=rotation,
                    rotation_mode="anchor",
                )

                bbox = ann_i.get_window_extent()
                bbox_data = bbox.transformed(polar_ax.transData.inverted())
                list_corners = bbox_data.corners()[:, 1]
                max_y = max(max_y, *list_corners)

        polar_ax.set_frame_on(False)
        polar_ax.xaxis.grid(False)
        polar_ax.yaxis.grid(False)
        polar_ax.set_xticks([])
        polar_ax.set_yticks([])
        axs_legend[i].axis("off")

    # Add the legend outside the plot
    for j, label in enumerate(vertices_labels):
        axs_legend[-1].scatter([], [], color=f"C{j}", s=100, label=label)
    axs_legend[-1].legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Archetypes",
        frameon=False,
        handlelength=1,
        handleheight=1,
    )

    # for ax_i in axs_polar:
    #     ax_i.set_ylim(0, max_y + 10)

    return ax
