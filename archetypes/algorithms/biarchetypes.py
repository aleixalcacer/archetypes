from math import inf

import numpy as np
from scipy.optimize import nnls
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_random_state

from .furthest_sum import furthest_sum


def _optimize_alphas(B, A):
    B = np.pad(B, ((0, 0), (0, 1)), "constant", constant_values=200)
    A = np.pad(A, ((0, 0), (0, 1)), "constant", constant_values=200)
    alphas = np.empty((B.shape[0], A.shape[0]))
    for j in range(alphas.T.shape[1]):
        alphas.T[:, j], _ = nnls(A.T, B.T[:, j])

    alphas /= alphas.sum(1)[:, None]
    alphas[np.isnan(alphas)] = 1 / alphas.shape[1]
    return alphas


def _optimize_betas(B, A):
    return _optimize_alphas(B, A)


def _optimize_gammas(B, A):
    B = np.pad(B, ((0, 1), (0, 0)), "constant", constant_values=200)
    A = np.pad(A, ((0, 1), (0, 0)), "constant", constant_values=200)

    gammas = np.empty((A.shape[1], B.shape[1]))
    for j in range(gammas.shape[1]):
        gammas[:, j], _ = nnls(A, B[:, j])

    gammas /= gammas.sum(0)[None, :]
    gammas[np.isnan(gammas)] = 1 / gammas.shape[0]
    return gammas


def _optimize_thetas(B, A):
    return _optimize_gammas(B, A)


def _biaa_simple(X, i_alphas, i_betas, i_gammas, i_thetas, max_iter, tol, verbose=False):
    alphas = i_alphas
    betas = i_betas
    gammas = i_gammas
    thetas = i_thetas

    Z = betas @ X @ thetas

    rss_0 = inf
    for n_iter in range(max_iter):
        if verbose and n_iter % 100 == 0:
            print(f"    Iteration: {n_iter + 1:{len(str(max_iter))}}, RSS: {rss_0:.2f}")

        alphas = _optimize_alphas(X, Z @ gammas)
        gammas = _optimize_gammas(X, alphas @ Z)
        Z = np.linalg.pinv(alphas) @ X @ np.linalg.pinv(gammas)
        betas = _optimize_betas(Z, X @ thetas)
        thetas = _optimize_thetas(Z, betas @ X)
        Z = betas @ X @ thetas
        rss = np.linalg.norm(X - alphas @ Z @ gammas)  # Frobenius norm
        if np.abs(rss_0 - rss) < tol:
            break
        rss_0 = rss

    return alphas, betas, gammas, thetas, rss_0, Z, n_iter


class BiAA(BaseEstimator):
    """
    Bi-Archetype Analysis estimator.

    Parameters
    ----------
    n_archetypes : tuple of int, default=(3, 2)
       The number of archetypes, both for samples and for features, to compute.
    n_init : int, default=5
        Number of time the archetype analysis algorithm will be run with
        different coefficient initialization. The final results will be the
        best output of *n_init* consecutive runs in terms of RSS.
    max_iter : int, default=300
       Maximum number of iterations of the archetype analysis algorithm
       for a single run.
    tol : float, default=1e-4
       Relative tolerance of two consecutive iterations to declare convergence.
    verbose : bool, default=False
       Verbosity mode.
    random_state : int, RandomState instance or None, default=None
       Determines random number generation of coefficients. Use an int to make
       the randomness deterministic.
    """

    def __init__(
        self,
        n_archetypes=(3, 2),
        n_init=5,
        max_iter=300,
        tol=1e-4,
        algorithm_init="auto",
        verbose=False,
        random_state=None,
    ):
        self.n_archetypes = n_archetypes
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.algorithm_init = algorithm_init
        self.verbose = verbose
        self.random_state = random_state

    def _check_data(self, X):
        if X.shape[0] < self.n_archetypes[0]:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_archetypes={self.n_archetypes[1]}."
            )
        if X.shape[1] < self.n_archetypes[1]:
            raise ValueError(
                f"n_samples={X.shape[1]} should be >= n_archetypes={self.n_archetypes[1]}."
            )

    def _check_parameters(self):
        if not isinstance(self.n_archetypes[0], int):
            raise TypeError
        if self.n_archetypes[0] <= 0:
            raise ValueError(f"n_archetypes[0] should be > 0, got {self.n_archetypes[0]} instead.")
        if not isinstance(self.n_archetypes[1], int):
            raise TypeError
        if self.n_archetypes[1] <= 0:
            raise ValueError(f"n_archetypes[1] should be > 0, got {self.n_archetypes[1]} instead.")

        if not isinstance(self.max_iter, int):
            raise TypeError
        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, got {self.max_iter} instead.")

        if not isinstance(self.n_init, int):
            raise TypeError
        if self.n_init <= 0:
            raise ValueError(f"n_int should be > 0, got {self.n_init} instead.")

        if not isinstance(self.algorithm_init, str):
            raise TypeError
        algorithm_init_names = ["auto", "random", "furthest_sum"]
        if self.algorithm_init not in algorithm_init_names:
            raise ValueError(
                f"algorithm_init must be one of {algorithm_init_names}, "
                f"got {self.algorithm_init} instead."
            )
        self._algorithm_init = self.algorithm_init

        if self._algorithm_init == "auto":
            self._algorithm_init = "furthest_sum"

        if not isinstance(self.verbose, bool):
            raise TypeError

    def _init_coefs(self, X, random_state):
        ind = random_state.choice(self.n_archetypes[0], X.shape[0])
        alphas = np.zeros((X.shape[0], self.n_archetypes[0]), dtype=np.float64)
        for i, j in enumerate(ind):
            alphas[i, j] = 1

        betas = np.zeros((self.n_archetypes[0], X.shape[0]), dtype=np.float64)
        if self._algorithm_init == "random":
            ind = random_state.choice(X.shape[0], self.n_archetypes)
        else:
            ind = furthest_sum(X.T, self.n_archetypes[0], random_state)
        for i, j in enumerate(ind):
            betas[i, j] = 1

        ind = random_state.choice(self.n_archetypes[1], X.shape[1])
        gammas = np.zeros((self.n_archetypes[1], X.shape[1]), dtype=np.float64)
        for j, i in enumerate(ind):
            gammas[i, j] = 1

        thetas = np.zeros((X.shape[1], self.n_archetypes[1]), dtype=np.float64)
        if self._algorithm_init == "random":
            ind = random_state.choice(X.shape[1], self.n_archetypes[1])
        else:
            ind = furthest_sum(X, self.n_archetypes[1], random_state)
        for j, i in enumerate(ind):
            thetas[i, j] = 1

        return alphas, betas, gammas, thetas

    def fit(self, X, y=None, **fit_params):
        """
        Compute Bi-Archetype Analysis.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to compute the archetypes.
            It must be noted that the data will be converted to C ordering,
            which will cause a memory copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_data(X, dtype=[np.float64, np.float32])
        self._check_parameters()
        self._check_data(X)
        random_state = check_random_state(self.random_state)

        self.rss_ = inf
        for i in range(self.n_init):
            if self.verbose:
                print(f"Initialization {i + 1:{len(str(self.n_init))}}/{self.n_init}")

            i_alphas, i_betas, i_gammas, i_thetas = self._init_coefs(X, random_state)

            alphas, betas, gammas, thetas, rss, Z, n_iter = _biaa_simple(
                X,
                i_alphas,
                i_betas,
                i_gammas,
                i_thetas,
                self.max_iter,
                self.tol,
                self.verbose,
            )

            if rss < self.rss_:
                self.alphas_ = alphas
                self.betas_ = betas
                self.gammas_ = gammas
                self.thetas_ = thetas
                self.archetypes_ = Z
                self.n_iter_ = n_iter
                self.rss_ = rss

        return self

    def transform(self, X):
        """
        Transform X to a biarchetype-distance space.

        In the new space, each dimension is the distance to the archetypes.
        Note that even if X is sparse, the array returned by `transform` will
        typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new_samples : ndarray of shape (n_samples, n_samp_archetypes)
            X samples transformed in the new space.
        X_new_features : ndarray of shape (n_feat_archetypes, n_features)
            X features transformed in the new space.
        """
        check_is_fitted(self)
        X = self._validate_data(X, dtype=[np.float64, np.float32])
        self._check_parameters()
        Z = self.archetypes_
        alphas = _optimize_alphas(X, Z @ self.gammas_)
        gammas = _optimize_gammas(X, self.alphas_ @ Z)
        return alphas, gammas

    def fit_transform(self, X, y=None, **fit_params):
        """
        Compute the biarchetypes and transform X to archetype-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new_samples : ndarray of shape (n_samples, n_samp_archetypes)
            X samples transformed in the new space.
        X_new_features : ndarray of shape (n_feat_archetypes, n_features)
            X features transformed in the new space.
        """
        return self.fit(X, y, **fit_params).transform(X)
