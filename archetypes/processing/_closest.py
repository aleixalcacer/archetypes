import numpy as np

from ._sort import sort_by_coefficients


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
