import numpy as np
import scipy
from opt_einsum import contract


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


def pmc(n_components, X, A, B):
    """
    Partitioning Around Components
    """
    components = B @ X
    components_index = np.argmax(B, axis=1)

    loss = np.linalg.norm(X - A @ components, "fro") ** 2
    combination = None
    for k in range(n_components):
        if k != 0:
            components[k - 1] = X[components_index[k - 1]]
        for i in range(X.shape[0]):
            components[k] = X[i]
            loss_i = np.linalg.norm(X - A @ components, "fro") ** 2
            if loss_i < loss:
                loss = loss_i
                combination = {"k": k, "i": i}

    if combination:
        components_index[combination["k"]] = combination["i"]

    B = np.zeros((n_components, X.shape[0]))
    B[np.arange(n_components), components_index] = 1

    return B
