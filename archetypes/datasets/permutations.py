import numpy as np

from ..utils import check_generator


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
    perms: list of array-like
        The permutations used to permute the dataset.
    """
    if perms is None:
        perms = [np.arange(s) for s in data.shape]

    # n = data.ndim

    for i, perms_i in enumerate(perms):
        data = np.swapaxes(data, 0, i)
        data = data[perms_i]
        data = np.swapaxes(data, 0, i)

    info = {"perms": perms}

    return data, info


def shuffle_dataset(data, generator=None):
    """Shuffle a dataset along each dimension.

    Parameters
    ----------
    data: array-like
        The dataset to shuffle.
    generator: int, Generator or None, default=None
        The generator to use for shuffling. If None, the default generator is used.

    Returns
    -------
    data: array-like
        The shuffled dataset.
    perms: list of array-like
        The permutations used to shuffle the dataset.
    """

    generator = check_generator(generator)

    perms = [np.arange(s) for s in data.shape]
    [generator.shuffle(indices_i) for indices_i in perms]

    data, info = permute_dataset(data, perms)

    return data, info


def sort_by_archetype_similarity(data, alphas):
    """Sort a dataset using the archetypal spaces previously computed.

    Parameters
    ----------
    data: array-like
        The dataset to sort.
    alphas: list of array-like
        The dataset in the archetypal spaces.

    Returns
    -------
    data: array-like
        The sorted dataset.
    perms: list of array-like
        The permutations used to sort the dataset.
    """

    values_to_sort = [(-np.max(a, axis=1), np.argmax(a, axis=1)) for a in alphas]
    # get index of ordered values
    perms = [np.lexsort(values_to_sort_i) for values_to_sort_i in values_to_sort]

    data, info = permute_dataset(data, perms)

    labels = [np.argmax(a, axis=1) for a in alphas]
    scores = [np.max(a, axis=1) for a in alphas]
    labels = [labels[i][perms[i]] for i in range(data.ndim)]
    scores = [scores[i][perms[i]] for i in range(data.ndim)]

    info["labels"] = labels
    info["scores"] = scores
    info["n_archetypes"] = [ai.shape[1] for ai in alphas]

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
    perms: list of array-like
        The permutations used to sort the dataset.
    """

    perms = [np.lexsort([labels_i]) for labels_i in labels]

    data, info = permute_dataset(data, perms)

    info["labels"] = labels

    return data, info
