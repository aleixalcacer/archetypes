import numpy as np
import pytest

from archetypes import NAA
from archetypes.datasets import make_archetypal_dataset
from archetypes.processing import (
    get_closest_n,
    get_closest_threshold,
    permute,
    shuffle,
    sort_by_coefficients,
    sort_by_labels,
)


@pytest.mark.parametrize(
    "shape,axis",
    [
        ((100, 100), None),
        ((50, 30), [0, 1]),
        ((50, 23, 15), 2),
        ((34, 24, 23), None),
    ],
)
def test_permute_shuffle(shape, axis):

    n_archetypes = 3

    # Set a random generator
    rng = np.random.default_rng(0)
    archetypes = rng.normal(
        size=[
            n_archetypes,
        ]
        * len(shape)
    )

    data, labels, perms = make_archetypal_dataset(archetypes, shape, generator=0)

    data_s, perms = shuffle(data, axis=axis)

    data_p, perms = permute(data, perms)

    assert np.allclose(data_p, data_s)


@pytest.mark.parametrize(
    "shape,arch_ndim",
    [
        [(10, 15, 24), 3],
        [(50, 30), 3],
        [(50,), 2],
        [(42, 35), 2],
        [(42, 35), 4],
    ],
)
def test_sort(shape, arch_ndim):
    n_archetypes = [
        3,
    ] * arch_ndim

    # Set a random generator
    rng = np.random.default_rng(0)
    archetypes = rng.normal(size=n_archetypes)

    data, _, _ = make_archetypal_dataset(archetypes, shape, noise=3, generator=0)

    assert data.ndim == archetypes.ndim

    for i in range(arch_ndim):
        if i < len(shape):
            assert data.shape[i] == shape[i]
        else:
            assert data.shape[i] == archetypes.shape[i]

    if len(shape) == arch_ndim:
        naa = NAA(n_archetypes=n_archetypes, max_iter=10, random_state=0)
        naa.fit(data)

        data_c, _, perms_c = sort_by_coefficients(data, naa.coefficients_)
        data_l, _, perms_l = sort_by_labels(data, naa.labels_)

        data_pc, _ = permute(data, perms_c)
        assert np.allclose(data_c, data_pc)

        data_pl, _ = permute(data, perms_l)
        assert np.allclose(data_l, data_pl)


@pytest.mark.parametrize(
    "shape,n,threshold",
    [
        [(100, 100), 0, 0.6],
        [(20, 450, 24), 4, 0.3],
        [(24, 32, 23), 8, 1],
        [(14, 15, 14, 16), 3, 0.5],
    ],
)
def test_get_closest(shape, n, threshold):
    n_archetypes = [
        3,
    ] * len(shape)

    # Set a random generator
    rng = np.random.default_rng(0)
    archetypes = rng.normal(size=n_archetypes)

    data, _, _ = make_archetypal_dataset(archetypes, shape, noise=3, generator=0)

    naa = NAA(n_archetypes=n_archetypes, max_iter=10, random_state=0)
    naa.fit(data)

    data_n, coeff_n, perms_n = get_closest_n(data, naa.coefficients_, n=n)
    data_t, coeff_t, perms_t = get_closest_threshold(data, naa.coefficients_, threshold=threshold)

    for a_s, s in zip(archetypes.shape, data_n.shape):
        assert a_s * n >= s

    for c in coeff_t:
        assert np.all(c.max(axis=1) >= threshold)
