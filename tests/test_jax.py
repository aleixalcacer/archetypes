import numpy as np
import pytest

from archetypes.jax import AA, BiAA

# create a test using pytest


@pytest.mark.parametrize(
    "method,method_kwargs",
    [
        ("autogd", None),
        ("autogd", {"optimizer": "sgd", "optimizer_params": {"learning_rate": 1e-3}}),
    ],
)
def test_AA(method, method_kwargs):
    shape = (100, 2)
    data = np.random.uniform(size=(shape))

    n_archetypes = 4
    model = AA(n_archetypes=n_archetypes, method=method, method_kwargs=method_kwargs)
    trans_data = model.fit_transform(data)

    model = AA(n_archetypes=n_archetypes, method=method, method_kwargs=method_kwargs)
    model = model.fit(data)
    trans_data = model.transform(data)

    assert trans_data.shape == (shape[0], n_archetypes)
    assert np.allclose(model.similarity_degree_.sum(axis=1), 1.0)
    assert np.allclose(trans_data.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "method,method_kwargs",
    [
        ("autogd", None),
        ("autogd", {"optimizer": "sgd", "optimizer_params": {"learning_rate": 1e-3}}),
    ],
)
def test_BiAA(method, method_kwargs):
    shape = (100, 100)
    data = np.random.uniform(size=(shape))

    n_archetypes = (4, 5)
    model = BiAA(n_archetypes=n_archetypes, method=method, method_kwargs=method_kwargs)
    trans_data = model.fit_transform(data)

    model = BiAA(n_archetypes=n_archetypes, method=method, method_kwargs=method_kwargs)
    model = model.fit(data)
    trans_data = model.transform(data)

    for i in range(len(n_archetypes)):
        assert trans_data[i].shape == (shape[i], n_archetypes[i])
        assert np.allclose(model.similarity_degree_[i].sum(axis=1), 1.0)
        assert np.allclose(trans_data[i].sum(axis=1), 1.0)
