from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_is_fitted, validate_data

from ._inits import aa_plus_plus, furthest_first, furthest_sum, uniform
from ._projection import l1_normalize_proj, unit_simplex_proj


class FairAA(TransformerMixin, BaseEstimator):
    """
    Fair Archetype Analysis.

    This class implements the Fair Archetype Analysis algorithm, which is a variant of the
    archetype analysis that incorporates fairness constraints.

    Parameters
    ----------
    n_archetypes: int
        The number of archetypes to compute.
    fairness_const: float, default=200.0
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
    method: str, default='pgd'
        The optimization method to use for the archetypes and the coefficients,
        must be one of the following: 'pgd', 'pseudo_pgd'. See :ref:`optimization-methods`.
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
        "method": [StrOptions({"pgd", "pseudo_pgd"})],
        "method_kwargs": [dict, None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        n_archetypes,
        *,
        fairness_const=200,
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
        self.fairness_const = fairness_const
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

    def fit(self, X, y=None, Z=None):
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
        Z : array-like of shape (n_samples, n_sensitive_features)
            Sensitive features to compute the fairness constraint.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.fit_transform(X, y, Z)
        return self

    def transform(self, X, Z):
        """
        Transform X to the archetypal space.

        In the new space, each dimension is the distance to the archetypes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
        Z : array-like of shape (n_samples, n_sensitive_features)
            Sensitive features to compute the fairness constraint.

        Returns
        -------
        A : ndarray of shape (n_samples, n_archetypes)
            X transformed in the new space.
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)
        X = np.ascontiguousarray(X)
        archetypes = self.archetypes_

        if self.n_archetypes_ == 1:
            n_samples = X.shape[0]
            return np.ones((n_samples, self.n_archetypes_), dtype=X.dtype)

        if self.method == "pgd":
            transform_func = pgd_transform
        elif self.method == "pseudo_pgd":
            transform_func = pseudo_pgd_transform

        method_kwargs = {} if self.method_kwargs is None else self.method_kwargs
        A = transform_func(
            X,
            Z,
            archetypes,
            fairness_const=self.fairness_const,
            max_iter=self.max_iter,
            tol=self.tol,
            **method_kwargs,
        )
        return A

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None, Z=None, **params):
        """
        Compute the archetypes and transform X to the archetypal space.

        Equivalent to fit(X).transform(X).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.
        Z : array-like of shape (n_samples, n_sensitive_features)
            Sensitive features to compute the fairness constraint.
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
            archetypes_ = np.mean(X, axis=0, keepdims=True)
            B_ = np.full((self.n_archetypes, n_samples), 1 / n_samples, dtype=X.dtype)
            A_ = np.ones((n_samples, self.n_archetypes), dtype=X.dtype)
            best_rss = squared_norm(X - archetypes_) + self.fairness_const * squared_norm(Z.T @ A_)
            n_iter_ = 0
            loss_ = [
                best_rss,
            ]

        else:
            if self.method == "pgd":
                fit_transform_func = pgd_fit_transform
            elif self.method == "pseudo_pgd":
                fit_transform_func = pseudo_pgd_fit_transform

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
                    Z,
                    archetypes,
                    fairness_const=self.fairness_const,
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


def pgd_transform(X, Z, archetypes, *, fairness_const, max_iter, tol, **kwargs):
    A = X @ np.linalg.pinv(archetypes)
    unit_simplex_proj(A)
    A, _, _, _, _, _ = _pgd_like_optimize_aa(
        X,
        A,
        None,
        Z,
        archetypes,
        fairness_const=fairness_const,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
        pseudo_pgd=False,
        update_B=False,
        **kwargs,
    )
    return A


def pgd_fit_transform(X, A, B, Z, archetypes, *, fairness_const, max_iter, tol, verbose, **kwargs):
    return _pgd_like_optimize_aa(
        X,
        A,
        B,
        Z,
        archetypes,
        fairness_const=fairness_const,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        pseudo_pgd=False,
        update_B=True,
        **kwargs,
    )


def pseudo_pgd_transform(X, Z, archetypes, *, fairness_const, max_iter, tol, **kwargs):
    A = X @ np.linalg.pinv(archetypes)
    l1_normalize_proj(A)
    A, _, _, _, _, _ = _pgd_like_optimize_aa(
        X,
        A,
        None,
        Z,
        archetypes,
        fairness_const=fairness_const,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
        pseudo_pgd=True,
        update_B=False,
        **kwargs,
    )
    return A


def pseudo_pgd_fit_transform(
    X, A, B, Z, archetypes, *, fairness_const, max_iter, tol, verbose, **kwargs
):
    return _pgd_like_optimize_aa(
        X,
        A,
        B,
        Z,
        archetypes,
        fairness_const=fairness_const,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        pseudo_pgd=True,
        update_B=True,
        **kwargs,
    )


def _pgd_like_optimize_aa(
    X,
    A,
    B,
    Z,
    archetypes,
    *,
    fairness_const,
    max_iter,
    tol,
    verbose=False,
    pseudo_pgd=False,
    update_B=True,
    step_size=1.0,
    max_iter_optimizer=10,
    beta=0.5,
    **kwargs,
):

    # precomputing and memory allocation
    BX = archetypes
    XXt = X @ X.T
    ABX = A @ BX
    ABX -= X
    XXtBt = X @ BX.T
    BXXtBt = BX @ BX.T
    AtXXt = A.T @ XXt

    ZZt = fairness_const * Z @ Z.T

    A_grad = np.empty_like(A)
    B_grad = np.empty_like(B)
    A_new = np.empty_like(A)
    B_new = np.empty_like(B)

    rss = squared_norm(ABX) + fairness_const * squared_norm(Z.T @ A)  # Now ABX stores ABX - X
    loss_list = [
        rss,
    ]

    step_size_A = step_size
    step_size_B = step_size

    for i in range(1, max_iter + 1):
        rss, step_size_A = _pgd_like_update_A_inplace(
            X,
            A,
            B,
            Z,
            BX,
            XXt,
            ABX,
            XXtBt,
            BXXtBt,
            ZZt,
            A_grad,
            A_new,
            pseudo_pgd,
            step_size_A,
            fairness_const,
            max_iter_optimizer,
            beta,
            rss,
        )

        if update_B:
            rss, step_size_B = _pgd_like_update_B_inplace(
                X,
                A,
                B,
                Z,
                BX,
                XXt,
                ABX,
                AtXXt,
                XXtBt,
                BXXtBt,
                B_grad,
                B_new,
                pseudo_pgd,
                step_size_B,
                fairness_const,
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
    Z,
    BX,
    XXt,
    ABX,
    XXtBt,
    BXXtBt,
    ZZt,
    A_grad,
    A_new,
    pseudo_pgd,
    step_size_A,
    fairness_const,
    max_iter_optimizer,
    beta,
    rss,
):
    # gradient wrt A
    A_grad = np.matmul(A, BXXtBt, out=A_grad)
    A_grad -= XXtBt
    A_grad += fairness_const * ZZt @ A
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
        ABX = np.matmul(A_new, BX, out=ABX)
        ABX -= X
        rss_new = squared_norm(ABX) + fairness_const * squared_norm(Z.T @ A_new)
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
        rss = rss_new

    return rss, step_size_A


def _pgd_like_update_B_inplace(
    X,
    A,
    B,
    Z,
    BX,
    XXt,
    ABX,
    AtXXt,
    XXtBt,
    BXXtBt,
    B_grad,
    B_new,
    pseudo_pgd,
    step_size_B,
    fairness_const,
    max_iter_optimizer,
    beta,
    rss,
):
    ABX += X
    B_grad = np.linalg.multi_dot([A.T, ABX, X.T], out=B_grad)
    B_grad -= np.matmul(A.T, XXt, out=AtXXt)
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
        ABX = np.linalg.multi_dot([A, B_new, X], out=ABX)
        ABX -= X
        rss_new = squared_norm(ABX) + fairness_const * squared_norm(Z.T @ A)
        improved = rss_new < rss
        if improved:
            step_size_B /= beta
            break
        step_size_B *= beta

    if improved:
        np.copyto(B, B_new)
        BX = np.matmul(B, X, out=BX)
        XXtBt = np.matmul(X, BX.T, out=XXtBt)
        BXXtBt = np.matmul(B, XXtBt, out=BXXtBt)
        rss = rss_new

    return rss, step_size_B


def verbose_print_rss(max_iter, rss, i):
    print(f"Iteration {i}/{max_iter}: RSS = {rss}")
