import numpy as np

from archetypes.utils import check_generator


def einsum(param_tensors, tensor):
    n = len(param_tensors)
    letters = [chr(i) for i in range(97, 97 + 2 * n)]
    inner_symbols = letters[:n]
    outer_symbols = letters[-n:]
    equation = [f"{o}{i}," for o, i in zip(outer_symbols, inner_symbols)]
    equation = "".join(equation) + "".join(inner_symbols) + "->" + "".join(outer_symbols)
    return np.einsum(equation, *param_tensors, tensor)


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

    n_archetypes = archetypes.shape

    generator = check_generator(generator)

    sizes = [
        generator.multinomial(size, np.repeat(1.0 / n, n)) for size, n in zip(shape, n_archetypes)
    ]

    labels = [
        np.hstack([np.repeat(val, rep) for val, rep in zip(range(n), size)])
        for size, n in zip(sizes, n_archetypes)
    ]

    A = [np.zeros((s_i, a_i)) for s_i, a_i in zip(shape, n_archetypes)]

    for A_i, labels_i in zip(A, labels):
        l_i_prev = -1
        for i, l_i in enumerate(labels_i):
            if l_i_prev != l_i:
                alpha_i = [0] * A_i.shape[1]
                alpha_i[l_i] = 1
                A_i[i, :] = alpha_i
                l_i_prev = l_i
            else:
                alpha_i = [alpha] * A_i.shape[1]
                alpha_i[l_i] = 1
                A_i[i, :] = generator.dirichlet(alpha_i)

    X = einsum(A, archetypes)

    # Add noise
    X += generator.normal(0, noise, size=shape)

    return X, labels
