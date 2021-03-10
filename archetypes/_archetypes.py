from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
import numpy as np
from math import inf
from scipy.optimize import nnls


def _optimize_alphas(X, Z):
    alphas = np.empty((X.shape[0], Z.shape[0]))
    for i in range(alphas.T.shape[1]):
        alphas.T[:, i], _ = nnls(Z.T, X.T[:, i])
    return alphas


def _optimize_betas(X, Z):
    return _optimize_alphas(Z, X)


def _aa_simple(X, archetypes, max_iter, tol):
    X = np.column_stack((X, np.full((X.shape[0],), 200)))
    Z = archetypes
    Z = np.column_stack((Z, np.full((Z.shape[0],), 200)))

    rss_0 = inf
    for n_iter in range(max_iter):
        alphas = _optimize_alphas(X, Z)
        Z = np.linalg.pinv(alphas) @ X
        betas = _optimize_betas(X, Z)
        Z = betas @ X
        rss = np.sum(np.power(X - alphas @ Z, 2))
        if np.abs(rss_0 - rss) < tol:
            break
        rss_0 = rss

    return alphas, betas, rss_0, Z[:, :-1], n_iter


class AA(BaseEstimator, TransformerMixin):
    def __init__(self, n_archetypes=4, n_init=5, max_iter=300, tol=1e-4, verbose=True, random_state=None):
        self.n_archetypes = n_archetypes
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state

    def _check_data(self, X):
        if X.shape[0] < self.n_archetypes:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_archetypes}."
            )

    def _check_parameters(self):
        if not isinstance(self.n_archetypes, int):
            raise TypeError
        if self.n_archetypes <= 0:
            raise ValueError(
                f"n_archetypes should be > 0, got {self.n_archetypes} instead."
            )

        if not isinstance(self.max_iter, int):
            raise TypeError
        if self.max_iter <= 0:
            raise ValueError(
                f"max_iter should be > 0, got {self.max_iter} instead."
            )

        if not isinstance(self.n_init, int):
            raise TypeError
        if self.n_init <= 0:
            raise ValueError(
                f"n_int should be > 0, got {self.n_init} instead."
            )

        if not isinstance(self.verbose, bool):
            raise TypeError

    def _init_archetypes(self, X, n_archetypes, random_state):
        ind = random_state.choice(X.shape[0], n_archetypes)
        archetypes = X[ind, :]
        return archetypes

    def fit(self, X, y=None, **fit_params):
        X = self._validate_data(X, dtype=[np.float64, np.float32])
        self._check_parameters()
        self._check_data(X)
        random_state = check_random_state(self.random_state)

        self.inertia_ = inf
        for i in range(self.n_init):
            archetypes = self._init_archetypes(X, self.n_archetypes, random_state)
            alphas, betas, inertia, Z, n_iter = _aa_simple(
                X, archetypes, self.max_iter, self.tol)

            if inertia < self.inertia_:
                self.alphas_ = alphas
                self.betas_ = betas
                self.archetypes_ = Z
                self.n_iter_ = n_iter
                self.inertia_ = inertia

        return self

    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, dtype=[np.float64, np.float32])
        self._check_parameters()

        X = np.column_stack((X, np.full((X.shape[0],), 200)))
        Z = self.archetypes_
        Z = np.column_stack((Z, np.full((Z.shape[0],), 200)))
        alphas = _optimize_alphas(X, Z)
        return alphas

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)
