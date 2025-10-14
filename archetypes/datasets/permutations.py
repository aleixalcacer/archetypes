import numpy as np
from sklearn.preprocessing import OneHotEncoder

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
        axis = list[np.arange(data.ndim)]
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


def sort_by_coefficients(data, coefficients):
    """Sort a dataset using the archetypal spaces previously computed.

    Parameters
    ----------
    data: array-like
        The dataset to sort.
    coefficients: list of array-like
        The coefficients to sort the dataset.
        archetypal space.

    Returns
    -------
    data: array-like
        The sorted dataset.
    coefficients: list of array-like
        The sorted coefficients.
    """
    data = np.asarray(data)
    coefficients = [np.asarray(alpha) for alpha in coefficients]

    ndim = len(coefficients)

    # reorder data by the number of elements in each 'archetypal group'
    perms = []
    for i, alpha_i in enumerate(coefficients):
        values, counts = np.unique(np.argmax(alpha_i, axis=1), return_counts=True)
        # sort values by counts
        perms_i = values[np.argsort(-counts)]
        # add missing indexes to perms
        perms_i = np.concatenate([perms_i, np.setdiff1d(np.arange(alpha_i.shape[1]), perms_i)])
        perms.append(perms_i)

    coefficients = [a_i[:, perms_i] for a_i, perms_i in zip(coefficients, perms)]

    values_to_sort = [(-np.max(a, axis=1), np.argmax(a, axis=1)) for a in coefficients]
    # get index of ordered values
    perms = [np.lexsort(values_to_sort_i) for values_to_sort_i in values_to_sort]

    data, _ = permute(data, perms)

    coefficients = [coefficients[i][perms[i]] for i in range(ndim)]

    return data, coefficients, perms


def sort_by_labels(data, labels):
    """Sort a dataset using the labels.

    Parameters
    ----------
    data: array-like
        The dataset to sort.
    labels: list of array-like
        The labels to sort the dataset.

    Returns
    -------
    data: array-like
        The sorted dataset.
    labels: list of array-like
        The sorted labels.
    """

    # encode labels as integers
    encoder = OneHotEncoder(sparse_output=False, dtype=int)
    labels = [np.asarray(label).reshape(-1, 1) for label in labels]
    coefficients = [encoder.fit_transform(label) for label in labels]

    data, coefficients, perms = sort_by_coefficients(data, coefficients)

    labels = [label[perm] for label, perm in zip(labels, perms)]

    return data, labels, perms


def get_closest_n(data, coefficients, n=10):
    """
    Return the n samples closest to each archetype based on their coefficients.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Input dataset containing all samples.
    coefficients: list of array-like
        List of coefficient matrices where each column represents the contribution of an archetype
        to a given sample. Samples with higher coefficients for an archetype are considered
        closer to that archetype.
    n : int, default=10
        Number of the closest samples to return for each archetype.

    Returns
    -------
    data : array-like
        The subset of the dataset containing the closest samples to each archetype.
    coefficients : list of array-like
        A list of the corresponding coefficient matrices for the returned samples.
    perms : list of array-like
        A list of the corresponding permutations for the returned samples.
    """

    data, coefficients, perms = sort_by_coefficients(data, coefficients)
    labels = [np.argmax(c, axis=1) for c in coefficients]

    idx = []
    for labels_i in labels:
        # Get the id of the first n occurrences of each label
        idx_i = []
        for label in set(labels_i):
            idx_i.extend(np.where(labels_i == label)[0][:n])

        idx_i = np.array(idx_i)
        if len(idx_i) == 0:
            data = np.array([], dtype=data.dtype).reshape([0] * data.ndim)
            coefficients = [
                np.array([], dtype=c.dtype).reshape(0, c.shape[1]) for c in coefficients
            ]
            perms = [[] for p in perms]
            return data, coefficients, perms

        idx.append(idx_i)

    data = data[np.ix_(*idx)]
    coefficients = [coeffs_i[idx_i] for coeffs_i, idx_i in zip(coefficients, idx)]
    perms = [perms_i[idx_i] for perms_i, idx_i in zip(perms, idx)]

    return data, coefficients, perms


def get_closest_threshold(data, coefficients, threshold=0.9):
    """
    Return the samples closest to each archetype based on their coefficients.

    Parameters
    ----------
    data : array-like
        Input dataset containing all samples.
    coefficients: list of array-like
        List of coefficient matrices where each column represents the contribution of an archetype
        to a given sample. Samples with higher coefficients for an archetype are considered
        closer to that archetype.
    threshold : float, default=0.9
        Minimum coefficient value for a sample to be considered close to an archetype.

    Returns
    -------
    data : array-like
        The subset of the dataset containing the closest samples to each archetype.
    coefficients : list of array-like
        A list of the corresponding coefficient matrices for the returned samples.
    perms : list of array-like
        A list of the corresponding permutations for the returned samples.
    """

    data, coefficients, perms = sort_by_coefficients(data, coefficients)
    labels = [np.argmax(c, axis=1) for c in coefficients]

    idx = []
    for labels_i, coeffs_i in zip(labels, coefficients):
        # Get the id of the occurrences of each label with coefficient above threshold
        idx_i = []
        for label in set(labels_i):
            c = np.where((labels_i == label) & (coeffs_i[:, label] >= threshold))[0]
            print(c)
            idx_i.extend(c)
        idx_i = np.array(idx_i)

        if len(idx_i) == 0:
            data = np.array([], dtype=data.dtype).reshape([0] * data.ndim)
            coefficients = [
                np.array([], dtype=c.dtype).reshape(0, c.shape[1]) for c in coefficients
            ]
            perms = [[] for p in perms]
            return data, coefficients, perms

        idx.append(idx_i)

    data = data[np.ix_(*idx)]
    coefficients = [coeffs_i[idx_i] for coeffs_i, idx_i in zip(coefficients, idx)]
    perms = [perms_i[idx_i] for perms_i, idx_i in zip(perms, idx)]

    return data, coefficients, perms
