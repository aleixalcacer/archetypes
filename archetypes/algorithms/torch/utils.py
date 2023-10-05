import torch


def einsum(param_tensors, tensor):
    n = len(param_tensors)
    letters = [chr(i) for i in range(97, 97 + 2 * n)]
    inner_symbols = letters[:n]
    outer_symbols = letters[-n:]
    equation = [f"{o}{i}," for o, i in zip(outer_symbols, inner_symbols)]
    equation = "".join(equation) + "".join(inner_symbols) + "->" + "".join(outer_symbols)
    return torch.einsum(equation, *param_tensors, tensor)


def hardmax(tensor, dim):
    id = [range(s) for s in tensor.shape]
    id[dim] = torch.argmax(tensor, dim=dim)
    tensor[...] = 0
    tensor[id] = 1
    return tensor


def softmax(tensor, dim):
    return torch.softmax(tensor, dim=dim)


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
