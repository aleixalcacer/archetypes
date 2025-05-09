from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_is_fitted

from ..utils import nnls, pmc
from ._aa import nnls_transform
from ._inits import aa_plus_plus, furthest_first, furthest_sum, uniform

# from ._projection import l1_normalize_proj, unit_simplex_proj


class ADA(TransformerMixin, BaseEstimator):
    """
    Archetypoid Analysis.

    Parameters
    ----------
    n_archetypes: int
        The number of archetypes to compute.
    max_iter : int, default=300
        Maximum number of iterations of the archetype analysis algorithm
        for a single run.
    tol : float, default=1e-4
        Relative tolerance of two consecutive iterations to declare convergence.
    init : str, default='uniform'
        Method used to initialize the archetypes, must be one of
        the following: 'uniform', 'furthest_sum', 'furthest_first' or 'aa_plus_plus'.
        See :ref:`initialization-methods`.
    n_init : int, default=1
        Number of time the archetype analysis algorithm will be run with different
        initializations. The final results will be the best output of n_init consecutive runs.
    init_kwargs : dict, default=None
        Additional keyword arguments to pass to the initialization method.
    save_init : bool, default=False
        If True, save the initial archetypes in the attribute `archetypes_init_`,
    method: str, default='nnls'
        The optimization method to use for the archetypes and the coefficients,
        must be one of the following: 'nnls', 'pgd', 'pseudo_pgd'. See :ref:`optimization-methods`.
    method_kwargs : dict, default=None
        Additional arguments to pass to the optimization method. See :ref:`optimization-methods`.
    verbose : bool, default=False
        Verbosity mode.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation of coefficients in initialization.
        Use an int to make the randomness reproducible.

    Attributes
    ----------
    archetypes_: np.ndarray
        The computed archetypes.
        It has shape `(n_archetypes, n_features)`.
    n_archetypes_: int
        The number of archetypes after fitting.
    archetypes_init_ : np.ndarray
        The initial archetypes. It is only available if `save_init=True`.
    similarity_degree_, A_ : np.ndarray
        The similarity degree of each sample to each archetype.
        It has shape `(n_samples, n_archetypes)`.
    archetypes_similarity_degree_, B_ : np.ndarray
        The similarity degree of each archetype to each sample.
        It has shape `(n_archetypes, n_samples)`.
    labels_ : np.ndarray
        The label of each sample. It is the index of the closest archetype.
        It has shape `(n_samples,)`.
    loss_ : list
        The loss at each iteration.
    rss_, reconstruction_error_ : float
        The residual sum of squares of the fitted data.

    References
    ----------
    """

    _parameter_constraints: dict = {
        "n_archetypes": [
            Interval(Integral, 1, None, closed="left"),
            # StrOptions({"auto"}),
        ],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "init": [
            StrOptions({"uniform", "furthest_sum", "furthest_first", "coreset", "aa_plus_plus"}),
            None,
        ],
        "init_kwargs": [dict, None],
        "save_init": [bool],
        "method": [StrOptions({"nnls"})],
        "method_kwargs": [dict, None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        n_archetypes,
        *,
        max_iter=300,
        tol=1e-4,
        init="uniform",
        n_init=1,
        init_kwargs=None,
        save_init=False,
        method="nnls",
        method_kwargs=None,
        verbose=False,
        random_state=None,
    ):
        self.n_archetypes = n_archetypes
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.init_kwargs = init_kwargs
        self.save_init = save_init
        self.method = method
        self.method_kwargs = method_kwargs
        self.verbose = verbose
        self.random_state = random_state

    def _check_params_vs_data(self, X):
        if X.shape[0] < self.n_archetypes:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_archetypes={self.n_archetypes}."
            )

    def _init_archetypes(self, X, rng):
        n_samples, n_features = X.shape

        if self.init == "uniform":
            init_archetype_func = uniform
        elif self.init == "furthest_sum":
            init_archetype_func = furthest_sum
        elif self.init == "furthest_first":
            init_archetype_func = furthest_first
        elif self.init == "aa_plus_plus":
            init_archetype_func = aa_plus_plus

        init_kwargs = {} if self.init_kwargs is None else self.init_kwargs
        B = np.zeros((self.n_archetypes, n_samples), dtype=X.dtype)
        ind = init_archetype_func(X, self.n_archetypes, random_state=rng, **init_kwargs)
        for i, j in enumerate(ind):
            B[i, j] = 1

        archetypes = X[ind]

        A = np.zeros((n_samples, self.n_archetypes), dtype=X.dtype)
        ind = rng.choice(self.n_archetypes, n_samples, replace=True)
        for i, j in enumerate(ind):
            A[i, j] = 1

        return A, B, archetypes

    def fit(self, X, y=None, **params):
        """
        Compute Archetype Analysis.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to compute the archetypes.
            It must be noted that the data will be converted to C ordering,
            which will cause a memory copy if the given data is not C-contiguous.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.fit_transform(X, y, **params)
        return self

    def transform(self, X):
        """
        Transform X to the archetypal space.

        In the new space, each dimension is the distance to the archetypes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        A : ndarray of shape (n_samples, n_archetypes)
            X transformed in the new space.
        """
        check_is_fitted(self)
        X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)
        X = np.ascontiguousarray(X)
        archetypes = self.archetypes_

        if self.n_archetypes_ == 1:
            n_samples = X.shape[0]
            return np.ones((n_samples, self.n_archetypes_), dtype=X.dtype)

        if self.method == "nnls":
            transform_func = nnls_transform
        elif self.method == "pgd":
            raise ValueError("pgd method is not supported for ADA.")
            # transform_func = pgd_transform
        elif self.method == "pseudo_pgd":
            raise ValueError("pseudo_pgd method is not supported for ADA.")
            # transform_func = pseudo_pgd_transform

        method_kwargs = {} if self.method_kwargs is None else self.method_kwargs
        A = transform_func(X, archetypes, max_iter=self.max_iter, tol=self.tol, **method_kwargs)
        return A

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        """
        Compute the archetypes and transform X to the archetypal space.

        Equivalent to fit(X).transform(X).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        A : ndarray of shape (n_samples, n_archetypes)
            X transformed in the new space.
        """
        X = self._validate_data(X, dtype=[np.float64, np.float32])
        self._check_params_vs_data(X)
        X = np.ascontiguousarray(X)

        if self.n_archetypes == 1:
            n_samples = X.shape[0]
            archetypes_ = np.mean(X, axis=0, keepdims=True)
            B_ = np.full((self.n_archetypes, n_samples), 1 / n_samples, dtype=X.dtype)
            A_ = np.ones((n_samples, self.n_archetypes), dtype=X.dtype)
            best_rss = squared_norm(X - archetypes_)
            n_iter_ = 0
            loss_ = [
                best_rss,
            ]

        else:
            if self.method == "nnls":
                fit_transform_func = nnls_ada_fit_transform
            elif self.method == "pgd":
                raise ValueError("pgd method is not supported for ADA.")
                # fit_transform_func = pgd_ada_fit_transform
            elif self.method == "pseudo_pgd":
                raise ValueError("pseudo_pgd method is not supported for ADA.")
                # fit_transform_func = pseudo_pgd_ada_fit_transform

            method_kwargs = {} if self.method_kwargs is None else self.method_kwargs

            rng = check_random_state(self.random_state)

            best_rss = np.inf
            for i in range(self.n_init):
                A, B, archetypes = self._init_archetypes(X, rng)

                if self.save_init:
                    self.B_init_ = B.copy()
                    self.archetypes_init_ = archetypes.copy()

                A, B, archetypes, n_iter, loss, _ = fit_transform_func(
                    X,
                    A,
                    B,
                    archetypes,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    verbose=self.verbose,
                    **method_kwargs,
                )

                rss = loss[-1]
                if i == 0 or rss < best_rss:
                    best_rss = rss
                    A_ = A
                    B_ = B
                    archetypes_ = archetypes
                    n_iter_ = n_iter
                    loss_ = loss

        self.A_ = A_
        self.B_ = B_
        self.archetypes_ = archetypes_
        self.n_iter_ = n_iter_
        self.loss_ = loss_
        self.rss_ = best_rss

        self.similarity_degree_ = self.A_
        self.archetypes_similarity_degree_ = self.B_
        self.n_archetypes_ = self.B_.shape[0]
        self.labels_ = np.argmax(self.A_, axis=1)

        # alias
        self.reconstruction_error_ = self.rss_

        return self.A_


def nnls_ada_fit_transform(X, A, B, archetypes, *, max_iter, tol, verbose, **kwargs):
    loss_list = [
        squared_norm(X - A @ archetypes),
    ]
    for i in range(1, max_iter + 1):
        A = nnls(X, archetypes, **kwargs)
        B = pmc(X, B, **kwargs)
        archetypes = np.matmul(B, X, out=archetypes)

        rss = squared_norm(X - A @ archetypes)
        convergence = abs(loss_list[-1] - rss) < tol
        loss_list.append(rss)
        if verbose and i % 10 == 0:  # Verbose mode (print RSS)
            verbose_print_rss(max_iter, rss, i)
        if convergence:
            break

    return A, B, archetypes, i, loss_list, convergence


def verbose_print_rss(max_iter, rss, i):
    print(f"Iteration {i}/{max_iter}: RSS = {rss}")
