# This code was derived from
# https://github.com/smair/archetypalanalysis-initialization/blob/main/code/baselines.py

import numpy as np

from ..utils import nnls


def uniform(X, k, random_state=None, **kwargs):
    # sample all points uniformly at random
    return random_state.choice(X.shape[0], k, replace=False)


def uniform_kernel(X, k, kernel, random_state=None, **kwargs):
    # sample all points uniformly at random
    return random_state.choice(X.shape[0], k, replace=False)


def furthest_first(X, k, random_state=None, **kwargs):
    n = X.shape[0]
    ind = []

    # sample first point uniformly at random
    i = random_state.choice(n, 1).item()
    ind.append(i)

    # sample remaining points
    for _ in range(k - 1):
        dist = np.array([np.linalg.norm(X - X[i], ord=2, axis=1) for i in ind])
        closest_cluster_id = dist.argmin(0)
        dist = dist[closest_cluster_id, np.arange(n)]
        # choose the point that is furthest away
        # from the points already chosen
        i = dist.argmax()
        ind.append(i)

    return ind


def furthest_sum(X, k, random_state=None, **kwargs):
    # Archetypal Analysis for Machine Learning
    # Morten Mørup and Lars Kai Hansen, 2010

    n = X.shape[0]
    ind = []

    # sample first point uniformly at random
    i = random_state.choice(n, 1).item()
    ind.append(i)

    # compute the (sum) of distances to the chosen point(s)
    # dist = np.sum((X-X[i])**2, axis=1)
    dist = np.linalg.norm(X - X[i], ord=2, axis=1)
    initial_dist = dist.copy()

    # chose k-1 points
    for _ in range(k - 1):
        # don't choose a chosen point again
        dist[ind] = 0.0
        # choose the point that is furthest away
        # to the sum of distances of points
        i = dist.argmax()
        ind.append(i)
        # add the distances to the new point to the current distances
        # dist = dist + np.sum((X-X[i])**2, axis=1)
        dist = dist + np.linalg.norm(X - X[i], ord=2, axis=1)

    # forget the first point chosen
    dist = dist - initial_dist
    ind = ind[1:]
    # don't choose a chosen point again
    dist[ind] = 0.0
    # chose another one
    i = dist.argmax()
    ind.append(i)

    return ind


def furthest_sum_kernel(X, k, kernel, random_state=None, **kwargs):
    # Archetypal Analysis for Machine Learning
    # Morten Mørup and Lars Kai Hansen, 2010

    n = X.shape[0]
    ind = []

    # compute the (sum) of distances to the chosen point(s)
    K = kernel(X, X, **kwargs)

    # Convert Gram matrix to distance
    dist = np.sqrt(-2 * K + np.diag(K)[:, None] + np.diag(K)[None, :])

    # sample first point uniformly at random
    i = random_state.choice(n, 1).item()
    ind.append(i)
    dist_i = dist[i].copy()
    initial_dist = dist_i.copy()

    # chose k-1 points
    for _ in range(k - 1):
        # don't choose a chosen point again
        dist_i[ind] = 0.0
        # choose the point that is furthest away
        # to the sum of distances of points
        i = dist_i.argmax()

        ind.append(i)
        # add the distances to the new point to the current distances
        dist_i = dist_i + dist[:, i]

    # forget the first point chosen
    dist_i = dist_i - initial_dist
    ind = ind[1:]
    # don't choose a chosen point again
    dist_i[ind] = 0.0
    # chose another one
    i = dist_i.argmax()
    ind.append(i)

    return ind


def coreset(X, k, random_state=None, **kwargs):
    # Coresets for Archetypal Analysis
    # Mair and Brefeld, 2019
    n = X.shape[0]
    dist = np.sum((X - X.mean(axis=0)) ** 2, axis=1)
    q = dist / dist.sum()
    ind = random_state.choice(n, k, p=q)
    return ind


def aa_plus_plus(X, k, random_state=None, kwargs=None):
    const = kwargs.get("const", 1_000)

    n = X.shape[0]
    ind = []

    # sample first point uniformly at random
    i = random_state.choice(n, 1).item()
    ind.append(i)

    # sample second point
    dist = np.sum((X - X[i]) ** 2, axis=1)
    i = random_state.choice(n, 1, p=dist / dist.sum()).item()
    ind.append(i)

    # sample remaining points
    for _ in range(k - 2):
        A = nnls(X, X[ind], const=const)
        dist = np.sum((X - A @ X[ind]) ** 2, axis=1)
        i = random_state.choice(n, 1, p=dist / dist.sum()).item()
        ind.append(i)

    return ind
