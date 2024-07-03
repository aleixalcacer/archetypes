import numpy as np

from ..utils import check_generator_numpy


def permute_dataset(data, perms=None) -> (np.array, dict):
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
    info: dict
        The information about the permutation.
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

    info = {"perms": perms}

    return data, info


def shuffle_dataset(data, ndim=None, generator=None):
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
    info: dict
        The information about the shuffling.
    """

    data = np.asarray(data)
    generator = check_generator_numpy(generator)

    perms = [np.arange(s) for s in data.shape[:ndim]]
    [generator.shuffle(indices_i) for indices_i in perms]

    data, info = permute_dataset(data, perms)

    return data, info


def sort_by_archetype_similarity(data, alphas, archetypes):
    """Sort a dataset using the archetypal spaces previously computed.

    Parameters
    ----------
    data: array-like
        The dataset to sort.
    alphas: list of array-like
        The dataset in the archetypal spaces.
    archetypes: list of array-like
        The archetypes.

    Returns
    -------
    data: array-like
        The sorted dataset.
    info: dict
        The information about the sorting.
    """
    data = np.asarray(data)
    alphas = [np.asarray(alpha) for alpha in alphas]
    archetypes = np.asarray(archetypes)

    ndim = len(alphas)

    # reorder data and archetypes by the number of elements in each 'archetypal group'
    perms = []
    for i, alpha_i in enumerate(alphas):
        values, counts = np.unique(np.argmax(alpha_i, axis=1), return_counts=True)
        # sort values by counts
        perms_i = values[np.argsort(-counts)]
        # add missing indexes to perms
        perms_i = np.concatenate([perms_i, np.setdiff1d(np.arange(alpha_i.shape[1]), perms_i)])
        perms.append(perms_i)

    alphas = [a_i[:, perms_i] for a_i, perms_i in zip(alphas, perms)]

    archetypes, _ = permute_dataset(archetypes, perms)

    values_to_sort = [(-np.max(a, axis=1), np.argmax(a, axis=1)) for a in alphas]
    # get index of ordered values
    perms = [np.lexsort(values_to_sort_i) for values_to_sort_i in values_to_sort]

    data, info = permute_dataset(data, perms)

    labels = [np.argmax(a, axis=1) for a in alphas]
    scores = [np.max(a, axis=1) for a in alphas]
    labels = [labels[i][perms[i]] for i in range(ndim)]
    scores = [scores[i][perms[i]] for i in range(ndim)]

    info["labels"] = labels
    info["scores"] = scores
    info["n_archetypes"] = [ai.shape[1] for ai in alphas]
    info["alphas"] = alphas
    info["archetypes"] = archetypes

    return data, info


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
    info: dict
        The information about the sorting.
    """

    data = np.asarray(data)
    labels = [np.asarray(label) for label in labels]

    ndim = len(labels)

    perms = [np.lexsort([labels_i]) for labels_i in labels]

    data, info = permute_dataset(data, perms)

    info["labels"] = [labels[i][perms[i]] for i in range(ndim)]

    return data, info
