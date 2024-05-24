import numpy as np
import scipy


def check_generator(generator=None):
    """Check if generator is an int, a Generator or None and return a Generator.

    Parameters
    ----------
    generator : int, Generator or None
        The generator to check. If None, the default generator is used.

    Returns
    -------
    generator : Generator
        The checked generator.
    """
    if generator is None:
        generator = np.random.default_rng()
    elif isinstance(generator, int):
        generator = np.random.default_rng(generator)
    elif isinstance(generator, np.random.Generator):
        pass
    else:
        raise ValueError(f"generator must be an int, a Generator or None, got {generator}")

    return generator


__all__ = ["check_generator"]


def nnls(B, A, max_iter=None, const=1000.0):
    """
    Non-negative Least Squares optimization.
    B = A @ X

    Parameters
    ----------
    B: np.ndarray
        The matrix to decompose.
    X: np.ndarray
        The matrix to decompose with.

    Returns
    -------
    A: np.ndarray
        The coefficients matrix.
    """
    B = np.hstack([B, const * np.ones((B.shape[0], 1))])
    A = np.hstack([A, const * np.ones((A.shape[0], 1))])

    X = np.zeros((B.shape[0], A.shape[0]))
    for i in range(B.shape[0]):
        X[i, :] = scipy.optimize.nnls(A.T, B[i, :], maxiter=max_iter)[0]

    X = np.maximum(X, 0)
    X = X / np.sum(X, axis=1)[:, None]

    return X
