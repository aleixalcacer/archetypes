# This code was derived from
# https://github.com/smair/archetypalanalysis-initialization/blob/main/code/baselines.py

import jax
import jax.numpy as jnp

from ..utils import nnls


def uniform(X, k, key, **kwargs):
    # sample all points uniformly at random
    key, subkey = jax.random.split(key)
    ind = jax.random.choice(subkey, X.shape[0], (k,), replace=False)
    return ind, key


def furthest_first(X, k, key, **kwargs):
    n = X.shape[0]
    ind = []

    # sample first point uniformly at random
    key, subkey = jax.random.split(key)
    i = jax.random.choice(subkey, n, (1,)).item()
    ind.append(i)

    # sample remaining points
    for _ in range(k - 1):
        dist = jnp.array([jnp.linalg.norm(X - X[i], ord=2, axis=1) for i in ind])
        closest_cluster_id = dist.argmin(0)
        dist = dist[closest_cluster_id, jnp.arange(n)]
        # choose the point that is furthest away
        # from the points already chosen
        i = dist.argmax()
        ind.append(i)

    return ind, key


def furthest_sum(X, k, key, **kwargs):
    # Archetypal Analysis for Machine Learning
    # Morten Mørup and Lars Kai Hansen, 2010

    n = X.shape[0]
    ind = []

    # sample first point uniformly at random
    key, subkey = jax.random.split(key)
    i = jax.random.choice(subkey, n, (1,)).item()
    ind.append(i)

    # compute the (sum) of distances to the chosen point(s)
    # dist = jnp.sum((X-X[i])**2, axis=1)
    dist = jnp.linalg.norm(X - X[i], ord=2, axis=1)
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
        # dist = dist + jnp.sum((X-X[i])**2, axis=1)
        dist = dist + jnp.linalg.norm(X - X[i], ord=2, axis=1)

    # forget the first point chosen
    dist = dist - initial_dist
    ind = ind[1:]
    # don't choose a chosen point again
    dist[ind] = 0.0
    # chose another one
    i = dist.argmax()
    ind.append(i)

    return ind, key


def coreset(X, k, key, **kwargs):
    # Coresets for Archetypal Analysis
    # Mair and Brefeld, 2019
    n = X.shape[0]
    dist = jnp.sum((X - X.mean(axis=0)) ** 2, axis=1)
    q = dist / dist.sum()
    key, subkey = jax.random.split(key)
    ind = jax.random.choice(subkey, n, k, p=q)
    return ind, key


def aa_plus_plus(X, k, key, kwargs=None):
    const = kwargs.get("const", 1_000)

    n = X.shape[0]
    ind = []

    # sample first point uniformly at random
    key, subkey = jax.random.split(key)
    i = jax.random.choice(subkey, n, (1,)).item()
    ind.append(i)

    # sample second point
    dist = jnp.sum((X - X[i]) ** 2, axis=1)
    key, subkey = jax.random.split(key)
    i = jax.random.choice(subkey, n, (1,), p=dist / dist.sum()).item()
    ind.append(i)

    # sample remaining points
    for _ in range(k - 2):
        A = nnls(X, X[ind], const=const)
        dist = jnp.sum((X - A @ X[ind]) ** 2, axis=1)
        key, subkey = jax.random.split(key)
        i = jax.random.choice(subkey, n, (1,), p=dist / dist.sum()).item()
        ind.append(i)

    return ind, key
