from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import validate_data

from ..utils import einsum
from ._inits import aa_plus_plus, furthest_first, furthest_sum, uniform
from ._projection import l1_normalize_proj, unit_simplex_proj


class SymmetricBiAA(TransformerMixin, BaseEstimator):
    """
    Symmetric BiArchetype Analysis.

    It is a variant of BiArchetype Analysis (BiAA) for square matrices
    where the rows and columns represent the same entities.

    It is particularly useful for analyzing similarity or distance matrices.

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
    init_params : dict, default=None
        Additional keyword arguments to pass to the initialization method.
    save_init : bool, default=False
        If True, save the initial archetypes in the attribute `archetypes_init_`,
    method: str, default='pgd'
        The optimization method to use for the archetypes and the coefficients,
        must be one of the following: pgd, pseudo_pgd'. See :ref:`optimization-methods`.
    method_params : dict, default=None
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
        It has shape (`n_archetypes`, `n_archetypes`).
    n_archetypes_: int
        The number of archetypes after fitting.
    archetypes_init_ : np.ndarray
        The initial archetypes. It is only available if `save_init=True`.
    coefficients_, A_ : np.ndarray
        The similarity degree of each sample to each archetype.
    arch_coefficients_, B_ : np.ndarray
        The similarity degree of each archetype to each sample.
    labels_ : np.ndarray
        The label of each sample. It is the index of the closest archetype.
    loss_ : list
        The loss at each iteration.
    rss_, reconstruction_error_ : float
        The residual sum of squares of the fitted data.

    References
    ----------
    """

    _parameter_constraints: dict = {
        "n_archetypes": [Interval(Integral, 1, None, closed="left"), tuple],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "init": [
            StrOptions({"uniform", "furthest_sum", "furthest_first", "coreset", "aa_plus_plus"}),
            None,
        ],
        "init_params": [dict, None],
        "save_init": [bool],
        "method": [
            StrOptions(
                {
                    "pgd",
                    "pseudo_pgd",
                }
            )
        ],
        "method_params": [dict, None],
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
        init_params=None,
        save_init=False,
        method="pgd",
        method_params=None,
        verbose=False,
        random_state=None,
    ):
        self.n_archetypes = n_archetypes
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.init_params = init_params
        self.save_init = save_init
        self.method = method
        self.method_params = method_params
        self.verbose = verbose
        self.random_state = random_state

    def _check_params_vs_data(self, X):
        # check X is square
        if X.shape[0] != X.shape[1]:
            raise ValueError(f"X should be a square matrix, got {X.shape}.")

        # check n_archetypes
        if X.shape[0] < self.n_archetypes:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_archetypes={self.n_archetypes}."
            )

    def _init_archetypes(self, X, rng):
        n_samples, _ = X.shape

        if self.init == "uniform":
            init_archetype_func = uniform
        elif self.init == "furthest_sum":
            init_archetype_func = furthest_sum
        elif self.init == "furthest_first":
            init_archetype_func = furthest_first
        elif self.init == "aa_plus_plus":
            init_archetype_func = aa_plus_plus

        init_params = {} if self.init_params is None else self.init_params

        B = np.zeros((self.n_archetypes, n_samples), dtype=X.dtype)
        ind = init_archetype_func(X, self.n_archetypes, random_state=rng, **init_params)
        for i, j in enumerate(ind):
            B[i, j] = 1

        archetypes = einsum([B, X, B.T])

        A = np.zeros((n_samples, self.n_archetypes), dtype=X.dtype)
        ind = rng.choice(self.n_archetypes, n_samples, replace=True)
        for i, j in enumerate(ind):
            A[i, j] = 1

        return A, B, archetypes

    def fit(self, X, y=None, **params):
        """
        Compute Symmetric BiArchetype Analysis.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_samples)
            Training instances to compute the archetypes. It should be a square matrix.
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
        X : array-like of shape (n_samples, n_samples)
            New data to transform.

        Returns
        -------
        A : ndarray of shape (n_samples, n_archetypes)
            X transformed in the new space.
        """
        raise NotImplementedError("BiAA does not support transform, use fit_transform instead.")

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        """
        Compute the archetypes and transform X to the archetypal space.

        Equivalent to fit(X).transform(X).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform. It must be noted that the data will be
            converted to C ordering, which will cause a memory copy if the given
            data is not C-contiguous.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        A : ndarray of shape (n_samples, n_archetypes)
            X transformed in the new space.
        """
        X = validate_data(self, X, dtype=[np.float64, np.float32])
        self._check_params_vs_data(X)
        X = np.ascontiguousarray(X)

        if self.n_archetypes == 1:
            n_samples = X.shape[0]

            archetypes_ = np.mean(X, axis=[0, 1], keepdims=True)
            B_ = np.full((self.n_archetypes, n_samples), 1 / n_samples, dtype=X.dtype)

            A_ = np.ones((n_samples, 1), dtype=X.dtype)

            best_rss = squared_norm(X - archetypes_)
            n_iter_ = 0
            loss_ = [
                best_rss,
            ]

        else:
            if self.method == "pgd":
                fit_transform_func = pgd_fit_transform
            elif self.method == "pseudo_pgd":
                fit_transform_func = pseudo_pgd_fit_transform

            method_params = {} if self.method_params is None else self.method_params

            rng = check_random_state(self.random_state)

            best_rss = np.inf
            for i in range(self.n_init):
                A, B, archetypes = self._init_archetypes(X, rng)

                if self.save_init:
                    self.B_init_ = [B[0].copy(), B[1].copy()]
                    self.archetypes_init_ = archetypes.copy()

                A, B, archetypes, n_iter, loss, _ = fit_transform_func(
                    X,
                    A,
                    B,
                    archetypes,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    verbose=self.verbose,
                    **method_params,
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

        self.coefficients_ = self.A_
        self.arch_coefficients_ = self.B_
        self.n_archetypes_ = self.archetypes_.shape
        self.labels_ = np.argmax(self.A_, axis=1)

        # alias
        self.reconstruction_error_ = self.rss_

        return self.A_


def pgd_transform(X, archetypes, *, max_iter, tol, **params):
    A = X @ np.linalg.pinv(archetypes)
    unit_simplex_proj(A)
    A, _, _, _, _, _ = _pgd_like_optimize_aa(
        X,
        A,
        None,
        archetypes,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
        pseudo_pgd=False,
        update_B=False,
        **params,
    )
    return A


def pgd_fit_transform(X, A, B, archetypes, *, max_iter, tol, verbose, **params):
    return _pgd_like_optimize_aa(
        X,
        A,
        B,
        archetypes,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        pseudo_pgd=False,
        update_B=True,
        **params,
    )


def pseudo_pgd_transform(X, archetypes, *, max_iter, tol, **params):
    A = X @ np.linalg.pinv(archetypes)
    l1_normalize_proj(A)
    A, _, _, _, _, _ = _pgd_like_optimize_aa(
        X,
        A,
        None,
        archetypes,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
        pseudo_pgd=True,
        update_B=False,
        **params,
    )
    return A


def pseudo_pgd_fit_transform(X, A, B, archetypes, *, max_iter, tol, verbose, **params):
    return _pgd_like_optimize_aa(
        X,
        A,
        B,
        archetypes,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        pseudo_pgd=True,
        update_B=True,
        **params,
    )


def _pgd_like_optimize_aa(
    X,
    A,
    B,
    archetypes,
    *,
    max_iter,
    tol,
    verbose=False,
    pseudo_pgd=False,
    update_B=True,
    step_size=1.0,
    max_iter_optimizer=10,
    beta=0.5,
    **params,
):

    A_grad = np.empty_like(A)
    B_grad = np.empty_like(B)
    A_new = np.empty_like(A)
    B_new = np.empty_like(B)

    rss = squared_norm(X - einsum([A, B, X, B.T, A.T]))
    loss_list = [
        rss,
    ]

    step_size_A = step_size
    step_size_B = step_size

    # Precomputed matrix
    AtA = einsum([A.T, A])
    AtXA = einsum([A.T, X, A])
    BXtBt = einsum([B, X.T, B.T])

    for i in range(1, max_iter + 1):
        rss, step_size_A = _pgd_like_update_A_inplace(
            X,
            A,
            B,
            AtA,
            AtXA,
            BXtBt,
            A_grad,
            A_new,
            pseudo_pgd,
            step_size_A,
            max_iter_optimizer,
            beta,
            rss,
        )

        if update_B:
            rss, step_size_B = _pgd_like_update_B_inplace(
                X,
                A,
                B,
                AtA,
                AtXA,
                BXtBt,
                B_grad,
                B_new,
                pseudo_pgd,
                step_size_B,
                max_iter_optimizer,
                beta,
                rss,
            )

        convergence = abs(loss_list[-1] - rss) < tol
        loss_list.append(rss)
        if verbose and i % 10 == 0:
            verbose_print_rss(max_iter, rss, i)
        if convergence:
            break

    return A, B, archetypes, i, loss_list, convergence


def _pgd_like_update_A_inplace(
    X,
    A,
    B,
    AtA,
    AtXA,
    BXtBt,
    A_grad,
    A_new,
    pseudo_pgd,
    step_size_A,
    max_iter_optimizer,
    beta,
    rss,
):
    BXBt = BXtBt.T

    A_grad = einsum([A, BXBt, AtA, BXtBt]) + einsum([A, BXtBt, AtA, BXBt])
    A_grad -= einsum([X, A, BXtBt]) + einsum([X.T, A, BXBt])

    if pseudo_pgd:
        # TODO: malloc here!
        A_grad -= np.expand_dims(np.einsum("ij,ij->i", A, A_grad), axis=1)
        project = l1_normalize_proj
    else:
        project = unit_simplex_proj

    # pgd & optimize step size
    # start with a large step size
    # then reduce the step size until we make any improvement wrt the loss
    improved = False
    for _ in range(max_iter_optimizer):
        A_new = np.multiply(-step_size_A, A_grad, out=A_new)
        A_new += A
        project(A_new)

        rss_new = squared_norm(X - einsum([A_new, BXtBt, A_new.T]))

        # if we make any improvement, break
        improved = rss_new < rss
        if improved:
            step_size_A /= beta  # leave some room for step size shrinkage
            break
            # if the new loss is worse, reduce the step size
        step_size_A *= beta

    # fix new A and update rss
    if improved:
        np.copyto(A, A_new)

        # update precomputed matrices
        AtA = np.matmul(A.T, A, out=AtA)
        AtX = np.matmul(A.T, X)
        AtXA = np.matmul(AtX, A, out=AtXA)

        rss = rss_new

    return rss, step_size_A


def _pgd_like_update_B_inplace(
    X,
    A,
    B,
    AtA,
    AtXA,
    BXtBt,
    B_grad,
    B_new,
    pseudo_pgd,
    step_size_B,
    max_iter_optimizer,
    beta,
    rss,
):

    BXBt = BXtBt.T

    B_grad = einsum([AtA, BXBt, AtA, B, X.T]) + einsum([AtA, BXtBt, AtA, B, X])
    B_grad -= einsum([AtXA, B, X.T]) + einsum([AtXA.T, B, X])

    if pseudo_pgd:
        # TODO: malloc here!
        B_grad -= np.expand_dims(np.einsum("ij,ij->i", B, B_grad), axis=1)
        project = l1_normalize_proj
    else:
        project = unit_simplex_proj

    # pgd & optimize step size
    improved = False
    for _ in range(max_iter_optimizer):
        B_new = np.multiply(-step_size_B, B_grad, out=B_new)
        B_new += B
        project(B_new)
        rss_new = squared_norm(X - einsum([A, B_new, X, B_new.T, A.T]))

        improved = rss_new < rss
        if improved:
            step_size_B /= beta
            break
        step_size_B *= beta

    if improved:
        np.copyto(B, B_new)

        # update precomputed matrices
        BXt = np.matmul(B, X.T)
        BXtBt = np.matmul(BXt, B.T, out=BXtBt)

        rss = rss_new

    return rss, step_size_B


def verbose_print_rss(max_iter, rss, i):
    print(f"Iteration {i}/{max_iter}: RSS = {rss}")
