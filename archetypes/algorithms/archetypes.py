from math import inf

import numpy as np
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, TransformerMixin
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


def _aa_simple(X, i_alphas, i_betas, max_iter, tol, verbose=False):
    alphas = i_alphas
    betas = i_betas

    Z = betas @ X

    rss_0 = inf
    for n_iter in range(max_iter):
        if verbose and n_iter % 100 == 0:
            print(f"    Iteration: {n_iter + 1:{len(str(max_iter))}}, RSS: {rss_0:.2f}")

        alphas = _optimize_alphas(X, Z)
        Z = np.linalg.pinv(alphas) @ X
        betas = _optimize_betas(Z, X)
        Z = betas @ X
        rss = np.linalg.norm(X - alphas @ Z)  # Frobenius norm
        if np.abs(rss_0 - rss) < tol:
            break
        rss_0 = rss

    return alphas, betas, rss_0, Z, n_iter


class AA(BaseEstimator, TransformerMixin):
    """
    Archetype Analysis estimator.

    Parameters
    ----------
    n_archetypes : int, default=4
        The number of archetypes to compute.
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

    References
    ----------
    .. [1] Adele Cutler, & Leo Breiman (1994). Archetypal analysis.
       Technometrics, 36, 338-347.


    """

    def __init__(
        self,
        n_archetypes=4,
        n_init=1,
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
        self.verbose = verbose
        self.random_state = random_state
        self.algorithm_init = algorithm_init

    def _check_data(self, X):
        if X.shape[0] < self.n_archetypes:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_archetypes={self.n_archetypes}."
            )

    def _check_parameters(self):
        if not isinstance(self.n_archetypes, int):
            raise TypeError
        if self.n_archetypes <= 0:
            raise ValueError(f"n_archetypes should be > 0, got {self.n_archetypes} instead.")

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
        ind = random_state.choice(self.n_archetypes, X.shape[0])
        alphas = np.zeros((X.shape[0], self.n_archetypes), dtype=np.float64)
        for i, j in enumerate(ind):
            alphas[i, j] = 1

        betas = np.zeros((self.n_archetypes, X.shape[0]), dtype=np.float64)
        if self._algorithm_init == "random":
            ind = random_state.choice(X.shape[0], self.n_archetypes)
        else:
            ind = furthest_sum(X.T, self.n_archetypes, random_state)
        for i, j in enumerate(ind):
            betas[i, j] = 1

        return alphas, betas

    def fit(self, X, y=None, **fit_params):
        """
        Compute Archetype Analysis.

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

            i_alphas, i_betas = self._init_coefs(X, random_state)

            alphas, betas, rss, Z, n_iter = _aa_simple(
                X, i_alphas, i_betas, self.max_iter, self.tol, self.verbose
            )

            if rss < self.rss_:
                self.alphas_ = alphas
                self.betas_ = betas
                self.archetypes_ = Z
                self.n_iter_ = n_iter
                self.rss_ = rss

        return self

    def transform(self, X):
        """
        Transform X to an archetype-distance space.

        In the new space, each dimension is the distance to the archetypes.
        Note that even if X is sparse, the array returned by `transform` will
        typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_archetypes)
            X transformed in the new space.
        """
        check_is_fitted(self)
        X = self._validate_data(X, dtype=[np.float64, np.float32])
        self._check_parameters()
        Z = self.archetypes_
        alphas = _optimize_alphas(X, Z)
        return alphas

    def fit_transform(self, X, y=None, **fit_params):
        """
        Compute the archetypes and transform X to archetype-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_archetypes)
            X transformed in the new space.
        """
        return self.fit(X, y, **fit_params).transform(X)
