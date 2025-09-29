import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, is_color_like, to_rgba_array


def create_palette(saturation, value, n_colors, int_colors=3):
    hue = np.linspace(0, 1, n_colors, endpoint=False)
    hue = np.hstack([hue[i::int_colors] for i in range(int_colors)])
    saturation = np.full(n_colors, saturation)
    value = np.full(n_colors, value)
    # convert to RGB
    c = mpl.colors.hsv_to_rgb(np.vstack([hue, saturation, value]).T)
    # Create palette
    palette = mpl.colors.ListedColormap(c)
    return palette


cmap = mpl.pyplot.get_cmap("viridis")  # default colormap for plots


def get_cmap():
    """Get the default colormap for plots."""
    return cmap


def set_cmap(name):
    """Set the default colormap for plots."""
    global cmap
    cmap = mpl.pyplot.get_cmap(name)
    return cmap


def map_colors(c, cmap="viridis", vmin=None, vmax=None):
    """
    Map values to RGBA colors like matplotlib.scatter(c=...).

    Parameters:
        c : array-like
            Numeric, categorical, or color-like values.
        cmap : str or Colormap
            Colormap name or object.
        vmin, vmax : float
            Min/max for numeric normalization.

    Returns:
        colors : np.ndarray of shape (N, 4)
    """
    c = np.atleast_1d(np.array(c))

    # Case 1: all color-like strings ("red", "#123456")
    if np.all([is_color_like(ci) for ci in c]):
        return to_rgba_array(c)

    # Case 2: numeric values
    elif np.issubdtype(c.dtype, np.number):
        if vmin is None:
            vmin = np.min(c)
        if vmax is None:
            vmax = np.max(c)
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        return sm.to_rgba(c)

    # Case 3: categorical / string values
    else:
        categories, indices = np.unique(c, return_inverse=True)
        n_colors = len(categories)
        cmap_obj = plt.get_cmap(cmap, n_colors)
        return cmap_obj(indices)
