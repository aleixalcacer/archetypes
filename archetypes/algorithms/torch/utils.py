import torch
from opt_einsum import contract

# from torch.nn.functional import gumbel_softmax


def einsum(param_tensors, tensor: torch.Tensor):
    n = len(param_tensors)
    diff = tensor.ndim - n

    letters = [chr(i) for i in range(97, 97 + 2 * n + diff)]
    inner_symbols = letters[:n] + letters[2 * n :]
    outer_symbols = letters[n : 2 * n] + letters[2 * n :]

    equation = [f"{o}{i}," for o, i in zip(outer_symbols[:n], inner_symbols[:n])]
    equation = "".join(equation) + "".join(inner_symbols) + "->" + "".join(outer_symbols)

    return contract(equation, *param_tensors, tensor)


def partial_einsum(param_tensors, tensor: torch.Tensor, index: []):
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


def einsum_dc(param_tensors, tensor):
    n = len(param_tensors)
    diff = tensor.ndim - n

    letters = [chr(i) for i in range(97, 97 + 2 * n + diff)]
    inner_symbols = letters[:n] + letters[2 * n :]

    equation = [f"{i}," for i in inner_symbols[:n]]
    equation = "".join(equation) + "".join(inner_symbols) + "->" + "".join(inner_symbols)

    return contract(equation, *param_tensors, tensor)


def hardmax(tensor, dim):
    id = [range(s) for s in tensor.shape]
    id[dim] = torch.argmax(tensor, dim=dim)
    tensor[...] = 0
    tensor[id] = 1
    return tensor


def softmax(tensor, dim):
    # return gumbel_softmax(tensor, tau=2, dim=dim, hard=False)
    y_soft = torch.softmax(tensor, dim)
    return y_soft


def normal_distance(X, maxoids):
    return ((X - maxoids) ** 2).sum(axis=-1)


def bernoulli_distance(X, maxoids):
    return -(X * maxoids.log() + (1 - X) * (1 - maxoids).log()).sum(axis=-1)


def poisson_distance(X, maxoids):
    return -(X * maxoids.log() - maxoids).sum(axis=-1)


def update_clusters_A(X, clusters, centroids, likelihood="normal"):
    new_clusters = torch.zeros_like(clusters)

    if likelihood == "bernoulli":
        e = 1e-8
        centroids[centroids == 0] = e
        centroids[centroids == 1] = 1 - e

    i = range(new_clusters.shape[0])
    j = globals()[f"{likelihood}_distance"](X[:, None, :], centroids[None, :, :]).argmin(
        axis=1
    )  # Optimized
    new_clusters[i, j] = 1

    return new_clusters


def update_clusters_D(X, clusters, maxoids, likelihood="normal"):
    return update_clusters_A(X.T, clusters.T, maxoids.T, likelihood).T


def loss_fun(X1, X2, loss="normal"):
    if loss == "normal":
        loss_i = torch.pow(X1 - X2, 2)
    elif loss == "bernoulli":
        e = 1e-8
        X2[X2 == 0] = e
        X2[X2 == 1] = 1 - e
        loss_i = -(X1 * X2.log() + (1 - X1) * (1 - X2).log())
    elif loss == "poisson":
        loss_i = -(X1 * torch.log(X2) - X2)
    else:
        raise ValueError("loss must be one of 'normal', 'bernoulli', 'poisson'")
    return loss_i
