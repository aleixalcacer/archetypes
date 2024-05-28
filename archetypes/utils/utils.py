import numpy as np
import scipy
from opt_einsum import contract


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


def nnls(B, A, max_iter=None, const=10.0):
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

    X = np.maximum(X, 1e-8)
    X = X / np.sum(X, axis=1)[:, None]

    return X


def einsum(params):
    n = len(params)
    letters = [chr(i) for i in range(97, 97 + n + 1)]
    symbols = letters
    equation = [f"{o}{i}," for o, i in zip(symbols[:n], symbols[1:])]
    equation = "".join(equation)[:-1] + "->" + f"{symbols[0]}{symbols[-1]}"
    return contract(equation, *params, optimize="auto")


def arch_einsum(param_tensors, tensor):
    n = len(param_tensors)
    diff = tensor.ndim - n

    letters = [chr(i) for i in range(97, 97 + 2 * n + diff)]
    inner_symbols = letters[:n] + letters[2 * n :]
    outer_symbols = letters[n : 2 * n] + letters[2 * n :]

    equation = [f"{o}{i}," for o, i in zip(outer_symbols[:n], inner_symbols[:n])]
    equation = "".join(equation) + "".join(inner_symbols) + "->" + "".join(outer_symbols)

    return contract(equation, *param_tensors, tensor, optimize="auto")


def partial_arch_einsum(param_tensors, tensor, index: []):
    n = len(param_tensors) + len(index)
    diff = tensor.ndim - n

    letters = [chr(i) for i in range(97, 97 + 2 * n + diff)]
    inner_symbols = letters[:n] + letters[2 * n :]
    outer_symbols = letters[n : 2 * n] + letters[2 * n :]
    res_equation = [
        f"{o}" if ind not in index else f"{i}"
        for ind, (o, i) in enumerate(zip(outer_symbols[:n], inner_symbols[:n]))
    ] + letters[2 * n :]

    equation = [
        f"{o}{i},"
        for ind, (o, i) in enumerate(zip(outer_symbols[:n], inner_symbols[:n]))
        if ind not in index
    ]
    equation = "".join(equation) + "".join(inner_symbols) + "->" + "".join(res_equation)

    return contract(equation, *param_tensors, tensor, optimize="auto")
