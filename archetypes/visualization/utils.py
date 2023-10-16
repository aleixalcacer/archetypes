import matplotlib as mpl
import numpy as np


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
