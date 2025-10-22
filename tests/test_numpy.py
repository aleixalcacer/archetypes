import numpy as np
import pytest
from scipy.sparse import csr_matrix

from archetypes import AA, ADA, BiAA
from archetypes.numpy._projection import l1_normalize_proj, unit_simplex_proj

# create a test using pytest


@pytest.mark.parametrize(
    "method,method_params",
    [
        ("nnls", None),
        ("nnls", {"max_iter_optimizer": 100, "const": 10}),
        ("pgd", None),
        ("pgd", {"max_iter_optimizer": 20, "beta": 0.8}),
        ("pseudo_pgd", None),
        ("pseudo_pgd", {"max_iter_optimizer": 20, "beta": 0.8}),
    ],
)
@pytest.mark.parametrize("data_format", ["dense", "sparse"])
def test_AA(method, method_params, data_format):
    shape = (100, 2)
    data = np.random.uniform(size=(shape))

    if data_format == "sparse":
        if method == "nnls":
            pytest.skip("nnls does not support sparse input")
        data = csr_matrix(data)

    n_archetypes = 4
    model = AA(n_archetypes=n_archetypes, method=method, method_params=method_params)
    trans_data = model.fit_transform(data)

    model = AA(n_archetypes=n_archetypes, method=method, method_params=method_params)
    model = model.fit(data)
    trans_data = model.transform(data)

    assert trans_data.shape == (shape[0], n_archetypes)
    assert np.allclose(model.coefficients_.sum(axis=1), 1.0)
    assert np.allclose(trans_data.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "method,method_params",
    [
        ("nnls", None),
        ("nnls", {"max_iter_optimizer": 100, "const": 10}),
        ("pgd", None),
        ("pgd", {"max_iter_optimizer": 20, "beta": 0.8}),
        ("pseudo_pgd", None),
        ("pseudo_pgd", {"max_iter_optimizer": 15, "beta": 0.7}),
    ],
)
@pytest.mark.parametrize("data_format", ["dense", "sparse"])
def test_BiAA(method, method_params, data_format):
    shape = (100, 100)
    data = np.random.uniform(size=(shape))

    if data_format == "sparse":
        if method == "nnls":
            pytest.skip("nnls does not support sparse input")
        data = csr_matrix(data)

    n_archetypes = (5, 4)
    model = BiAA(n_archetypes=n_archetypes, method=method, method_params=method_params)
    trans_data = model.fit_transform(data)

    model = BiAA(n_archetypes=n_archetypes, method=method, method_params=method_params)
    model = model.fit(data)

    for i in range(len(n_archetypes)):
        assert trans_data[i].shape == (shape[i], n_archetypes[i])
        assert np.allclose(model.coefficients_[i].sum(axis=1), 1.0)
        assert np.allclose(trans_data[i].sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "method,method_params",
    [
        ("pgd", None),
        ("pgd", {"max_iter_optimizer": 20, "beta": 0.8}),
        ("pseudo_pgd", None),
        ("pseudo_pgd", {"max_iter_optimizer": 20, "beta": 0.8}),
    ],
)
@pytest.mark.parametrize(
    "shape,n_archetypes",
    [
        ((200, 300), (3, 4)),
        ((10, 8, 6), (2, 3, 2)),
    ],
)
def test_NAA(method, method_params, shape, n_archetypes):
    from archetypes import NAA

    data = np.random.uniform(size=(shape))

    model = NAA(n_archetypes=n_archetypes, method=method, method_params=method_params)
    trans_data = model.fit_transform(data)

    model = NAA(n_archetypes=n_archetypes, method=method, method_params=method_params)
    model = model.fit(data)

    for i in range(len(n_archetypes)):
        assert trans_data[i].shape == (shape[i], n_archetypes[i])
        assert np.allclose(model.coefficients_[i].sum(axis=1), 1.0)
        assert np.allclose(trans_data[i].sum(axis=1), 1.0)


@pytest.mark.parametrize("proj_func", [unit_simplex_proj, l1_normalize_proj])
def test_projection(proj_func):
    x = np.random.uniform(size=(100, 4))
    proj_func(x)
    assert np.all(x >= 0)
    assert np.all(x <= 1)
    assert np.allclose(x.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "method,method_params",
    [
        ("nnls", None),
        ("nnls", {"max_iter_optimizer": 100, "const": 10}),
    ],
)
def test_ADA(method, method_params):
    shape = (100, 2)
    data = np.random.uniform(size=(shape))

    n_archetypes = 4
    model = ADA(n_archetypes=n_archetypes, method=method, method_params=method_params)
    trans_data = model.fit_transform(data)

    model = ADA(n_archetypes=n_archetypes, method=method, method_params=method_params)
    model = model.fit(data)
    trans_data = model.transform(data)

    assert trans_data.shape == (shape[0], n_archetypes)
    assert np.allclose(model.coefficients_.sum(axis=1), 1.0)
    assert np.allclose(trans_data.sum(axis=1), 1.0)
