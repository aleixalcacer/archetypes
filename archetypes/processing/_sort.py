import numpy as np
from sklearn.preprocessing import OneHotEncoder

from ._permute import permute


def sort_by_coefficients(data, coefficients, reorder=False):
    """Sort a dataset using the archetypal spaces previously computed.

    Parameters
    ----------
    data: array-like
        The dataset to sort.
    coefficients: list of array-like
        The coefficients to sort the dataset.
        archetypal space.
    order: bool, optional
        Whether to order the archetypal groups by size.
        Default is False.

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

    if reorder:
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


def sort_by_labels(data, labels, reorder=False):
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

    data, coefficients, perms = sort_by_coefficients(data, coefficients, reorder=reorder)

    labels = [label[perm] for label, perm in zip(labels, perms)]

    return data, labels, perms
