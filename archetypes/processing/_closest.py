import numpy as np

from ._sort import sort_by_coefficients


def get_closest_n(data, coefficients, n=10, reorder=False):
    """
    Return the n samples closest to each archetype based on their coefficients.

    Parameters
    ----------
    data : array-like of shape
        Input dataset containing all samples.
    coefficients: list of array-like
        List of coefficient matrices where each column represents the contribution of an archetype
        to a given sample. Samples with higher coefficients for an archetype are considered
        closer to that archetype.
    n : int, tuple, default=10
        Number of the closest samples to return for each archetype.
        if tuple, it should be of the same length as the number of coefficient matrices.
        Default is 10 for all archetypes in all dimensions.

    reorder : bool, optional
        Whether to reorder the archetypal groups by size.
        Default is False.

    Returns
    -------
    data : array-like
        The subset of the dataset containing the closest samples to each archetype.
    coefficients : list of array-like
        A list of the corresponding coefficient matrices for the returned samples.
    perms : list of array-like
        A list of the corresponding permutations for the returned samples.
    """

    data, coefficients, perms = sort_by_coefficients(data, coefficients, reorder=reorder)
    labels = [np.argmax(c, axis=1) for c in coefficients]
    if isinstance(n, int):
        n = [n] * len(coefficients)
    elif len(n) != len(coefficients):
        raise ValueError("n should be an int or a tuple of the same length as coefficients")

    idx = []
    for i, labels_i in enumerate(labels):
        # Get the id of the first n occurrences of each label
        idx_i = []
        for label in set(labels_i):
            idx_i.extend(np.where(labels_i == label)[0][: n[i]])

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


def get_closest_threshold(data, coefficients, threshold=0.9, reorder=False):
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
    threshold : float, tuple, default=0.9
        Minimum coefficient value for a sample to be considered close to an archetype.
    reorder : bool, optional
        Whether to reorder the archetypal groups by size.
        Default is False.

    Returns
    -------
    data : array-like
        The subset of the dataset containing the closest samples to each archetype.
    coefficients : list of array-like
        A list of the corresponding coefficient matrices for the returned samples.
    perms : list of array-like
        A list of the corresponding permutations for the returned samples.
    """

    data, coefficients, perms = sort_by_coefficients(data, coefficients, reorder=reorder)
    labels = [np.argmax(c, axis=1) for c in coefficients]
    if isinstance(threshold, (int, float)):
        threshold = [threshold] * len(coefficients)
    elif len(threshold) != len(coefficients):
        raise ValueError(
            "threshold should be a float or a tuple of the same length as coefficients"
        )

    idx = []
    for i, (labels_i, coeffs_i) in enumerate(zip(labels, coefficients)):
        # Get the id of the occurrences of each label with coefficient above threshold
        idx_i = []
        for label in set(labels_i):
            c = np.where((labels_i == label) & (coeffs_i[:, label] >= threshold[i]))[0]
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
