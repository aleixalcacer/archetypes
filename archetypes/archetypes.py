from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
import numpy as np
from math import inf
from scipy.optimize import nnls


def _optimize_alphas(B, A):
    B = np.pad(B, ((0, 0), (0, 1)), 'constant', constant_values=200)
    A = np.pad(A, ((0, 0), (0, 1)), 'constant', constant_values=200)
    alphas = np.empty((B.shape[0], A.shape[0]))
    for j in range(alphas.T.shape[1]):
        alphas.T[:, j], _ = nnls(A.T, B.T[:, j])
    alphas /= alphas.sum(1)[:, None]
    alphas[np.isnan(alphas)] = 1 / alphas.shape[1]
    return alphas

def _optimize_betas(B, A):
    return _optimize_alphas(B, A)


def _aa_simple(X, i_alphas, i_betas, max_iter, tol, verbose=False):
    alphas = i_alphas
    betas = i_betas

    Z = betas @ X

    rss_0 = inf
    for n_iter in range(max_iter):
        if verbose:
            if n_iter % 100 == 0:
                print(f"    Iteration: {n_iter + 1:{len(str(max_iter))}}, RSS: {rss_0:.2f}")

        alphas = _optimize_alphas(X, Z)
        Z = np.linalg.pinv(alphas) @ X
        betas = _optimize_betas(Z, X)
        Z = betas @ X
        rss = np.sum(np.power(X - alphas @ Z, 2))
        if np.abs(rss_0 - rss) < tol:
            break
        rss_0 = rss

    return alphas, betas, rss_0, Z, n_iter


class AA(BaseEstimator, TransformerMixin):
    def __init__(self, n_archetypes=4, n_init=5, max_iter=300, tol=1e-4, verbose=False, random_state=None):
        self.n_archetypes = n_archetypes
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state

    def _check_data(self, X):
        if X.shape[0] < self.n_archetypes:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_archetypes={self.n_archetypes}."
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

    def _init_coefs(self, X, random_state):
        ind = random_state.choice(self.n_archetypes, X.shape[0])
        alphas = np.zeros((X.shape[0], self.n_archetypes), dtype=np.float64)
        for i, j in enumerate(ind):
            alphas[i, j] = 1

        ind = random_state.choice(X.shape[0], self.n_archetypes)
        betas = np.zeros((self.n_archetypes, X.shape[0]), dtype=np.float64)
        for i, j in enumerate(ind):
            betas[i, j] = 1

        return alphas, betas

    def fit(self, X, y=None, **fit_params):
        X = self._validate_data(X, dtype=[np.float64, np.float32])
        self._check_parameters()
        self._check_data(X)
        random_state = check_random_state(self.random_state)

        self.rss_ = inf
        for i in range(self.n_init):
            if self.verbose:
                print(f"Initialization {i + 1:{len(str(self.n_init))}}/{self.n_init}")

            i_alphas, i_betas = self._init_coefs(X, random_state)

            alphas, betas, rss, Z, n_iter = _aa_simple(
                X, i_alphas, i_betas, self.max_iter, self.tol, self.verbose)

            if rss < self.rss_:
                self.alphas_ = alphas
                self.betas_ = betas
                self.archetypes_ = Z
                self.n_iter_ = n_iter
                self.rss_ = rss

        return self

    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, dtype=[np.float64, np.float32])
        self._check_parameters()
        Z = self.archetypes_
        alphas = _optimize_alphas(X, Z)
        return alphas

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)
