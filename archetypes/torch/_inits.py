# This code was derived from
# https://github.com/smair/archetypalanalysis-initialization/blob/main/code/baselines.py

import torch

from ..utils import nnls


def uniform(X, k, generator: torch.Generator = None, **kwargs):
    # sample all points uniformly at random
    return torch.randperm(X.shape[0], generator=generator)[:k]


def furthest_first(X, k, generator: torch.Generator = None, **kwargs):
    n = X.shape[0]
    ind = []

    # sample first point uniformly at random
    i = torch.randint(0, n, (1,), generator=generator).item()
    ind.append(i)

    # sample remaining points
    for _ in range(k - 1):
        dist = []
        for i in ind:
            d = torch.linalg.norm(X - X[i], ord=2, dim=1)
            dist.append(d)

        dist = torch.stack(dist)

        closest_cluster_id = dist.argmin(0)
        dist = dist[closest_cluster_id, torch.arange(n)]
        # choose the point that is furthest away
        # from the points already chosen
        i = dist.argmax()
        ind.append(i)

    return ind


def furthest_sum(X, k, generator: torch.Generator = None, **kwargs):
    # Archetypal Analysis for Machine Learning
    # Morten Mørup and Lars Kai Hansen, 2010

    n = X.shape[0]
    ind = []

    # sample first point uniformly at random
    i = torch.randint(0, n, (1,), generator=generator).item()
    ind.append(i)

    # compute the (sum) of distances to the chosen point(s)
    # dist = torch.sum((X-X[i])**2, dim=1)
    dist = torch.linalg.norm(X - X[i], ord=2, dim=1)
    initial_dist = dist.clone()

    # chose k-1 points
    for _ in range(k - 1):
        # don't choose a chosen point again
        dist[torch.tensor(ind)] = 0.0
        # choose the point that is furthest away
        # to the sum of distances of points
        i = dist.argmax()
        ind.append(i)
        # add the distances to the new point to the current distances
        # dist = dist + torch.sum((X-X[i])**2, dim=1)
        dist = dist + torch.linalg.norm(X - X[i], ord=2, dim=1)

    # forget the first point chosen
    dist = dist - initial_dist
    ind = ind[1:]
    # don't choose a chosen point again
    dist[torch.tensor(ind)] = 0.0
    # chose another one
    i = dist.argmax()
    ind.append(i)

    return ind


def coreset(X, k, generator: torch.Generator = None, **kwargs):
    # Coresets for Archetypal Analysis
    # Mair and Brefeld, 2019
    # n = X.shape[0]
    dist = torch.sum((X - X.mean(dim=0)) ** 2, dim=1)
    q = dist / dist.sum()
    ind = torch.multinomial(q, k, replacement=True, generator=generator)
    return ind


def aa_plus_plus(X, k, generator: torch.Generator = None, kwargs=None):
    const = kwargs.get("const", 1_000)

    n = X.shape[0]
    ind = []

    # sample first point uniformly at random
    i = torch.randint(0, n, (1,), generator=generator).item()
    ind.append(i)

    # sample second point
    dist = torch.sum((X - X[i]) ** 2, dim=1)
    i = torch.multinomial(dist / dist.sum(), 1, replacement=True, generator=generator).item()
    ind.append(i)

    # sample remaining points
    for _ in range(k - 2):
        A = torch.tensor(nnls(X, X[ind], const=const), dtype=X.dtype, device=X.device)
        dist = torch.sum((X - A @ X[ind]) ** 2, dim=1)
        i = torch.multinomial(dist / dist.sum(), 1, replacement=True, generator=generator).item()
        ind.append(i)

    return ind
