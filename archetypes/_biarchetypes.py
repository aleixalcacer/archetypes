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

def _optimize_alphas(B, A):
    B = np.pad(B, ((0, 0), (0, 1)), 'constant', constant_values=200)
    A = np.pad(A, ((0, 0), (0, 1)), 'constant', constant_values=200)

    alphas = np.empty((B.shape[0], A.shape[0]))
    for j in range(alphas.T.shape[1]):
        alphas.T[:, j], _ = nnls(A.T, B.T[:, j])

    return alphas

def _optimize_betas(B, A):
    return _optimize_alphas(B, A)


def _optimize_gammas(B, A):
    B = np.pad(B, ((0, 1), (0, 0)), 'constant', constant_values=200)
    A = np.pad(A, ((0, 1), (0, 0)), 'constant', constant_values=200)
    gammas = np.empty((A.shape[1], B.shape[1]))
    for j in range(gammas.shape[1]):
        gammas[:, j], _ = nnls(A, B[:, j])
    return gammas

def _optimize_thetas(B, A):
    return _optimize_gammas(B, A)


def _biaa_simple(X, init_alphas, init_betas, init_gammas, init_thetas, max_iter, tol):
    alphas = init_alphas
    betas = init_betas
    gammas = init_gammas
    thetas = init_thetas
    
    Z = betas @ X @ thetas

    rss_0 = inf
    for n_iter in range(max_iter):
        alphas = _optimize_alphas(X, Z @ gammas)
        gammas = _optimize_gammas(X, alphas @ Z)

        Z = np.linalg.pinv(alphas) @ X @ np.linalg.pinv(gammas)
        betas = _optimize_betas(Z, X @ thetas)
        thetas = _optimize_thetas(Z, betas @ X)

        Z = betas @ X @ thetas
        rss = np.sum(np.power(X - alphas @ Z @ gammas, 2))
        if np.abs(rss_0 - rss) < tol:
            break
        rss_0 = rss

    return alphas, betas, gammas, thetas, rss_0, Z, n_iter


class BiAA(BaseEstimator, TransformerMixin):
    def __init__(self, k=4, c=3, n_init=5, max_iter=300, tol=1e-4, verbose=True,
                 random_state=None):
        self.k = k
        self.c = c
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state

    def _check_data(self, X):
        if X.shape[0] < self.k:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.k}."
            )
        if X.shape[1] < self.c:
            raise ValueError(
                f"n_samples={X.shape[1]} should be >= n_clusters={self.c}."
            )

    def _check_parameters(self):
        if not isinstance(self.k, int):
            raise TypeError
        if self.k <= 0:
            raise ValueError(
                f"c should be > 0, got {self.k} instead."
            )
        if self.c <= 0:
            raise ValueError(
                f"c should be > 0, got {self.k} instead."
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

    def _init_coefs(self, X, k, c, random_state):
        ind = random_state.choice(X.shape[0], k)
        betas = np.zeros((k, X.shape[0]), dtype=np.float64)
        for i, j in enumerate(ind):
            betas[i, j] = 1

        ind = random_state.choice(X.shape[1], c)
        thetas = np.zeros((X.shape[1], c), dtype=np.float64)
        for j, i in enumerate(ind):
            thetas[i, j] = 1

        ind = random_state.choice(k, X.shape[0])
        alphas = np.zeros((X.shape[0], k), dtype=np.float64)
        for i, j in enumerate(ind):
            alphas[i, j] = 1

        ind = random_state.choice(c, X.shape[1])
        gammas = np.zeros((c, X.shape[1]), dtype=np.float64)
        for j, i in enumerate(ind):
            gammas[i, j] = 1

        return alphas, betas, gammas, thetas

    def fit(self, X, y=None, **fit_params):
        X = self._validate_data(X, dtype=[np.float64, np.float32])
        self._check_parameters()
        self._check_data(X)
        random_state = check_random_state(self.random_state)

        self.inertia_ = inf
        for i in range(self.n_init):
            i_alphas, i_betas, i_gammas, i_thetas = self._init_coefs(X, self.k, self.c, random_state)

            alphas, betas, gammas, thetas, inertia, Z, n_iter = _biaa_simple(
                X, i_alphas, i_betas, i_gammas, i_thetas,
                self.max_iter, self.tol)

            if inertia < self.inertia_:
                self.alphas_ = alphas
                self.betas_ = betas
                self.gammas_ = gammas
                self.thetas_ = thetas
                self.archetypes_ = Z
                self.n_iter_ = n_iter
                self.inertia_ = inertia

        return self

    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, dtype=[np.float64, np.float32])
        self._check_parameters()
        Z = self.archetypes_

        alphas = _optimize_alphas(X, Z @ self.gammas_)
        gammas = _optimize_gammas(X, self.alphas_ @ Z)

        return alphas, gammas

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)
