import numpy as np

from ..utils import check_generator_numpy, partial_arch_einsum
from .permutations import sort_by_labels


def make_archetypal_dataset(
    archetypes, shape, alpha=1.0, noise=0.0, generator=None
) -> (np.array, list):
    """
    Generate a dataset from archetypes.

    Parameters
    ----------
    archetypes : np.ndarray
        The archetypes.
    shape : tuple of int
        The shape of the dataset.
    alpha : float, default=1.
        The concentration parameter of the Dirichlet distribution.
    noise : float, default=0.
        The standard deviation of the gaussian noise.
    generator : int, Generator instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    np.ndarray
        The dataset.
    list of np.ndarray
        The labels for each dimension.
    """
    ndim = len(shape)

    n_archetypes = archetypes.shape[:ndim]

    generator = check_generator_numpy(generator)

    sizes = [
        generator.multinomial(size, np.repeat(1.0 / n, n)) for size, n in zip(shape, n_archetypes)
    ]

    labels = [
        np.hstack([np.repeat(val, rep) for val, rep in zip(range(n), size)])
        for size, n in zip(sizes, n_archetypes)
    ]

    new_labels = [np.zeros(size) for size in shape]

    A = [np.zeros((s_i, a_i)) for s_i, a_i in zip(shape, n_archetypes)]

    for A_i, labels_i, n_archetypes_i, new_labels_i in zip(A, labels, n_archetypes, new_labels):
        for k in range(n_archetypes_i):
            idx = np.where(labels_i == k)[0]
            alpha_i = [alpha] * n_archetypes_i
            alpha_i[k] = 1.0
            A_i[idx, :] = generator.dirichlet(alpha_i, size=len(idx))
            # Reassign labels to the argmax of the archetypes
            new_labels_i[idx] = A_i[idx].argmax(axis=1)

    # Generate the dataset
    X = partial_arch_einsum(A, archetypes, index=[])

    # Add noise
    X += generator.normal(0, noise, size=X.shape)

    # Sort the dataset by the labels
    X, info = sort_by_labels(X, new_labels)

    return X, info["labels"]
