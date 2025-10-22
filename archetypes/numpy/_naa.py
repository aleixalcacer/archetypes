from numbers import Integral, Real

import numpy as np
from pykronecker import KroneckerProduct as Kron
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import validate_data

from archetypes.utils import arch_einsum, unfold

from ._inits import aa_plus_plus, furthest_first, furthest_sum, uniform
from ._projection import l1_normalize_proj, unit_simplex_proj


class NAA(TransformerMixin, BaseEstimator):
    """
    N-dimensional archetype Analysis.

    Parameters
    ----------
    n_archetypes: tuple
        The number of archetypes to compute along each dimension.
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
        Number of times the archetype analysis algorithm will be run with different
        initializations. The final results will be the best output of n_init consecutive runs.
    init_params : dict, default=None
        Additional keyword arguments to pass to the initialization method.
    save_init : bool, default=False
        If True, save the initial archetypes in the attribute `archetypes_init_`,
    method: str, default='pgd'
        The optimization method to use for the archetypes and the coefficients,
        must be one of the following: 'nnls, pgd, pseudo_pgd'. See :ref:`optimization-methods`.
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
        It has shape `n_archetypes`.
    n_archetypes_: tuple
        The number of archetypes after fitting.
    archetypes_init_ : np.ndarray
        The initial archetypes. It is only available if `save_init=True`.
    coefficients_, A_ : list of np.ndarray
        The similarity degree of each sample to each archetype.
    arch_coefficients_, B_ : list of np.ndarray
        The similarity degree of each archetype to each sample.
    labels_ : list of np.ndarray
        The label of each sample. It is the index of the closest archetype.
    loss_ : list
        The loss at each iteration.
    rss_, reconstruction_error_ : float
        The residual sum of squares of the fitted data.

    References
    ----------
    """

    _parameter_constraints: dict = {
        "n_archetypes": [tuple, list],
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
        for i in range(len(self.n_archetypes)):
            if X.shape[i] < self.n_archetypes[i]:
                raise ValueError(
                    f"n_samples[{i}]={X.shape[i]} should be "
                    f">= n_archetypes[{i}]={self.n_archetypes[i]}."
                )

    def _init_archetypes(self, X, rng):

        if self.init == "uniform":
            init_archetype_func = uniform
        elif self.init == "furthest_sum":
            init_archetype_func = furthest_sum
        elif self.init == "furthest_first":
            init_archetype_func = furthest_first
        elif self.init == "aa_plus_plus":
            init_archetype_func = aa_plus_plus

        init_params = {} if self.init_params is None else self.init_params

        B = []

        for i in range(len(self.n_archetypes)):
            B_i = np.zeros((self.n_archetypes[i], X.shape[i]), dtype=X.dtype)
            ind = init_archetype_func(
                unfold(X, i), self.n_archetypes[i], random_state=rng, **init_params
            )
            for i, j in enumerate(ind):
                B_i[i, j] = 1
            B.append(B_i)

        archetypes = arch_einsum(B, X)

        A = []
        for i in range(len(self.n_archetypes)):
            A_i = np.zeros((X.shape[i], self.n_archetypes[i]), dtype=X.dtype)
            ind = rng.choice(self.n_archetypes[i], X.shape[i], replace=True)
            for j, k in enumerate(ind):
                A_i[j, k] = 1
            A.append(A_i)

        return A, B, archetypes

    def fit(self, X, y=None, **params):
        """
        Compute Archetype Analysis.

        Parameters
        ----------
        X : array-like of shape (n_samples_0, n_samples_1)
            Training instances to compute the archetypes. Must be a sparse matrix in CSR format.
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
            It must be noted that the data will be converted to C ordering,
            which will cause a memory copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it’s not in CSR format.

        Returns
        -------
        A : ndarray of shape (n_samples, n_archetypes)
            X transformed in the new space.
        """
        raise NotImplementedError("NiAA does not support transform, use fit_transform instead.")

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        """
        Compute the archetypes and transform X to the archetypal space.

        Equivalent to fit(X).transform(X).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
            It must be noted that the data will be converted to C ordering,
            which will cause a memory copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it’s not in CSR format.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        A : ndarray of shape (n_samples, n_archetypes)
            X transformed in the new space.
        """

        X = validate_data(self, X, dtype=[np.float64, np.float32], allow_nd=True)
        self._check_params_vs_data(X)

        X = np.ascontiguousarray(X)

        if self.n_archetypes == (1, 1):

            archetypes_ = np.mean(X, keepdims=True)

            B_ = [
                np.full((self.n_archetypes[i], X.shape[i]), 1 / X.shape[i], dtype=X.dtype)
                for i in range(len(self.n_archetypes))
            ]

            A_ = [np.ones((X.shape[i], 1), dtype=X.dtype) for i in range(len(self.n_archetypes))]

            best_rss = squared_norm(X - arch_einsum(A_, archetypes_))
            n_iter_ = 0
            loss_ = [
                best_rss,
            ]

        else:
            if self.method == "nnls":
                raise ValueError("NAA does not support nnls method.")
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
                    self.B_init_ = [b.copy() for b in B]
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
        self.labels_ = [np.argmax(self.A_[0], axis=1), np.argmax(self.A_[1], axis=1)]

        # alias
        self.reconstruction_error_ = self.rss_

        return self.A_


def normal_loss(X, A, B):
    return squared_norm(X - arch_einsum(A, arch_einsum(B, X)))


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

    A_grad = [np.empty_like(a) for a in A]
    B_grad = [np.empty_like(b) for b in B]
    A_new = [np.empty_like(a) for a in A]
    B_new = [np.empty_like(b) for b in B]

    C = [a @ b for a, b in zip(A, B)]

    # TODO: Explore ways to avoid recomputing all gradients on each iteration.
    # Previous attempts made it slower, but there may be more efficient approaches.

    rss = normal_loss(X, A, B)

    loss_list = [
        rss,
    ]

    step_size_A = [step_size for _ in A]
    step_size_B = [step_size for _ in B]

    for i in range(1, max_iter + 1):
        rss, step_size_A = _pgd_like_update_A_inplace(
            X,
            A,
            B,
            C,
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
                C,
                B_grad,
                B_new,
                pseudo_pgd,
                step_size_B,
                max_iter_optimizer,
                beta,
                rss,
            )

        archetypes = arch_einsum(B, X)

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
    C,
    A_grad,
    A_new,
    pseudo_pgd,
    step_size_A,
    max_iter_optimizer,
    beta,
    rss,
):

    for i, A_i in enumerate(A):
        Xn = unfold(X, i)
        # Mn = kron_except(C, i)  # Kronecker de los modos distintos a n
        Mn = Kron([C[j] for j in range(len(C)) if j != i])
        MnXntBt = Mn @ (Xn.T @ B[i].T)
        A_grad[i] = -Xn @ MnXntBt + C[i] @ (Xn @ (Mn.T @ MnXntBt))

        if pseudo_pgd:
            # TODO: malloc here!
            A_grad[i] -= np.expand_dims(np.einsum("ij,ij->i", A[i], A_grad[i]), axis=1)
            project = l1_normalize_proj
        else:
            project = unit_simplex_proj

        # pgd & optimize step size
        # start with a large step size
        # then reduce the step size until we make any improvement wrt the loss
        improved = False
        for _ in range(max_iter_optimizer):
            A_new[i] = np.multiply(-step_size_A[i], A_grad[i], out=A_new[i])
            A_new[i] += A_i
            project(A_new[i])

            A_temp = [a if j != i else A_new[i] for j, a in enumerate(A)]
            rss_new = normal_loss(X, A_temp, B)

            # if we make any improvement, break
            improved = rss_new < rss
            if improved:
                step_size_A[i] /= beta  # leave some room for step size shrinkage
                break
                # if the new loss is worse, reduce the step size
            step_size_A[i] *= beta

        # fix new A and update rss
        if improved:
            np.copyto(A[i], A_new[i])
            C[i] = A[i] @ B[i]
            rss = rss_new

    return rss, step_size_A


def _pgd_like_update_B_inplace(
    X,
    A,
    B,
    C,
    B_grad,
    B_new,
    pseudo_pgd,
    step_size_B,
    max_iter_optimizer,
    beta,
    rss,
):

    for i, B_i in enumerate(B):
        Xn = unfold(X, i)
        # Mn = kron_except(C, i)  # Kronecker de los modos distintos a n
        Mn = Kron([C[j] for j in range(len(C)) if j != i])
        B_grad[i] = ((-A[i].T @ Xn) @ Mn) @ Xn.T + ((((A[i].T @ C[i]) @ Xn) @ Mn.T) @ Mn) @ Xn.T

        if pseudo_pgd:
            # TODO: malloc here!
            B_grad[i] -= np.expand_dims(np.einsum("ij,ij->i", B[i], B_grad[i]), axis=1)
            project = l1_normalize_proj
        else:
            project = unit_simplex_proj

        # pgd & optimize step size
        improved = False
        for _ in range(max_iter_optimizer):
            B_new[i] = np.multiply(-step_size_B[i], B_grad[i], out=B_new[i])
            B_new[i] += B_i
            project(B_new[i])

            B_temp = [b if j != i else B_new[i] for j, b in enumerate(B)]
            rss_new = normal_loss(X, A, B_temp)

            improved = rss_new < rss
            if improved:
                step_size_B[i] /= beta
                break
            step_size_B[i] *= beta

        if improved:
            np.copyto(B[i], B_new[i])
            C[i] = A[i] @ B[i]
            rss = rss_new

    return rss, step_size_B


def verbose_print_rss(max_iter, rss, i):
    print(f"Iteration {i}/{max_iter}: RSS = {rss}")
