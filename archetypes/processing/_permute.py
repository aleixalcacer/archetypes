import numpy as np

from ..utils import check_generator_numpy


def permute(data, perms=None):
    """Permute a dataset along each dimension.

    Parameters
    ----------
    data: array-like
        The dataset to permute.
    perms: list of array-like
        The permutations to use. If None, no permutation is applied.

    Returns
    -------
    data: array-like
        The permuted dataset.
    perms: list of array-like
        The permutations used.
    """
    data = np.asarray(data)

    if perms is None:
        perms = [np.arange(s) for s in data.shape]
    else:
        perms = [np.asarray(perm) for perm in perms]

    # n = data.ndim

    for i, perms_i in enumerate(perms):
        data = np.swapaxes(data, 0, i)
        data = data[perms_i]
        data = np.swapaxes(data, 0, i)

    return data, perms


def shuffle(data, axis=None, generator=None):
    """Shuffle a dataset along each dimension.

    Parameters
    ----------
    data: array-like
        The dataset to shuffle.
    ndim: list of int or None, default=None
        The dimensions to shuffle. If None, all dimensions are shuffled.
    generator: int, Generator or None, default=None
        The generator to use for shuffling. If None, the default generator is used.

    Returns
    -------
    data: array-like
        The shuffled dataset.
    perms: list of array-like
        The permutations used.
    """

    data = np.asarray(data)
    generator = check_generator_numpy(generator)

    if axis is None:
        axis = list(np.arange(data.ndim))
    elif isinstance(axis, int):
        axis = [axis]
    elif isinstance(axis, (list, tuple)):
        # Check if some element of the axis is greater than data.ndim
        if any(a >= data.ndim for a in axis):
            raise ValueError(f"axis elements must be less than {data.ndim}, got {axis}")
    else:
        raise ValueError(f"axis type {type(axis)} is not supported, use int, list of tuple of ints")

    perms = [np.arange(s) for s in data.shape]
    [generator.shuffle(perms_i) for i, perms_i in enumerate(perms) if i in axis]

    data, perms = permute(data, perms)

    return data, perms
