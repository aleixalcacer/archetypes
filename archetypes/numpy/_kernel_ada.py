from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.metrics.pairwise import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import validate_data

from ._inits import furthest_sum_kernel, uniform_kernel
from ._projection import l1_normalize_proj, unit_simplex_proj


def precomputed_kernel(X, X2):
    return X


class KernelADA(TransformerMixin, BaseEstimator):
    """
    Kernel Archetypoid Analysis.

    This class implements the Kernel Archetypoid Analysis algorithm, which is a variant of the
    archetypoid analysis that uses kernel methods to compute the archetypoids.

    Parameters
    ----------
    n_archetypes: int
        The number of archetypes to compute.
    kernel : str, default=rbf
        The kernel to use for the archetype analysis, must be one of
        the following: 'linear', 'poly', 'rbf', 'sigmoid'.
    kernel_params : dict, default=None
        Additional keyword arguments to pass to the kernel function.
    max_iter : int, default=300
        Maximum number of iterations of the archetype analysis algorithm
        for a single run.
    tol : float, default=1e-4
        Relative tolerance of two consecutive iterations to declare convergence.
    init : str, default='uniform'
        Method used to initialize the archetypes, must be one of
        the following: 'uniform' or 'furthest_sum'.
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
        must be one of the following: 'pgd'. See :ref:`optimization-methods`.
    method_params : dict, default=None
        Additional arguments to pass to the optimization method. See :ref:`optimization-methods`.
    batch_size : int, default=None
        The batch size to use when updating the archetypes. If None, all data is used at once.
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
    coefficients_, A_ : np.ndarray
        The similarity degree of each sample to each archetype.
        It has shape `(n_samples, n_archetypes)`.
    arch_coefficients_, B_ : np.ndarray
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
        "kernel": [
            StrOptions({"linear", "poly", "rbf", "sigmoid", "precomputed"}),
            None,
        ],
        "kernel_params": [dict, None],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "init": [
            StrOptions({"uniform", "furthest_sum"}),
            None,
        ],
        "init_params": [dict, None],
        "save_init": [bool],
        "method": [StrOptions({"pgd", "pseudo_pgd"})],
        "method_params": [dict, None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        n_archetypes,
        *,
        kernel="rbf",
        kernel_params=None,
        max_iter=300,
        tol=1e-4,
        init="uniform",
        n_init=1,
        init_params=None,
        save_init=False,
        method="pgd",
        method_params=None,
        batch_size=None,
        verbose=False,
        random_state=None,
    ):
        self.n_archetypes = n_archetypes
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.init_params = init_params
        self.save_init = save_init
        self.method = method
        self.method_params = method_params
        self.batch_size = batch_size
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
            init_archetype_func = uniform_kernel
        elif self.init == "furthest_sum":
            init_archetype_func = furthest_sum_kernel

        init_params = {} if self.init_params is None else self.init_params
        kernel_params = {} if self.kernel_params is None else self.kernel_params

        # concatenate kernel params with init params
        init_params = {**kernel_params, **init_params}

        B = np.zeros((self.n_archetypes, n_samples), dtype=X.dtype)
        ind = init_archetype_func(
            X, self.n_archetypes, eval(f"{self.kernel}_kernel"), random_state=rng, **init_params
        )
        for i, j in enumerate(ind):
            B[i, j] = 1

        archetypes = X[ind]

        A = np.zeros((n_samples, self.n_archetypes), dtype=X.dtype)
        ind = rng.choice(self.n_archetypes, n_samples, replace=True)
        for i, j in enumerate(ind):
            A[i, j] = 1

        return A, B, archetypes

    def fit(self, X, y=None, weights=None, **params):
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
        weights: np.ndarray, default=None
            Sample weights.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.fit_transform(X, y, weights, **params)
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

        raise NotImplementedError("Transform method is not implemented yet.")

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None, weights=None, **params):
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
        X = validate_data(self, X, dtype=[np.float64, np.float32])
        self._check_params_vs_data(X)
        X = np.ascontiguousarray(X)

        if weights is not None:
            weights = np.asarray(weights)
            if weights.ndim != 1 or weights.shape[0] != X.shape[0]:
                raise ValueError(
                    f"Weights must be a 1D array of shape (n_samples,), got {weights.shape}."
                )
        else:
            weights = np.ones(X.shape[0], dtype=X.dtype)

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
            if self.method == "pgd":
                fit_transform_func = pgd_fit_transform
            elif self.method == "pseudo_pgd":
                fit_transform_func = pseudo_pgd_fit_transform

            method_params = {} if self.method_params is None else self.method_params

            rng = check_random_state(self.random_state)

            # Compute kernel

            if self.kernel == "linear":
                ZXXt = linear_kernel(X)
            elif self.kernel == "poly":
                ZXXt = polynomial_kernel(X)
            elif self.kernel == "rbf":
                ZXXt = rbf_kernel(X)
            elif self.kernel == "sigmoid":
                ZXXt = sigmoid_kernel(X)
            elif self.kernel == "precomputed":
                ZXXt = X
            else:
                raise ValueError(f"Unknown kernel: {self.kernel}")

            self.X_ = (
                X.copy()
            )  # Store the original data for computing the kernel matrix in transform

            ZXXt = ZXXt.copy()

            W = np.diag(weights)

            best_rss = np.inf
            for i in range(self.n_init):
                A, B, _ = self._init_archetypes(X, rng)

                if self.save_init:
                    self.B_init_ = B.copy()
                    # self.archetypes_init_ = archetypes.copy()

                A, B, _, n_iter, loss, _ = fit_transform_func(
                    X,
                    A,
                    B,
                    W,
                    ZXXt,
                    ZXXt,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    batch_size=self.batch_size,
                    verbose=self.verbose,
                    **method_params,
                )

                rss = loss[-1]
                if i == 0 or rss < best_rss:
                    best_rss = rss
                    A_ = A
                    B_ = B
                    n_iter_ = n_iter
                    loss_ = loss

        self.A_ = A_
        self.B_ = B_
        self.archetypes_ = None if self.kernel == "precomputed" else (X.T @ B_.T).T
        self.n_iter_ = n_iter_
        self.loss_ = loss_
        self.rss_ = best_rss

        self.coefficients_ = self.A_
        self.arch_coefficients_ = self.B_
        self.n_archetypes_ = self.B_.shape[0]
        self.labels_ = np.argmax(self.A_, axis=1)

        # alias
        self.reconstruction_error_ = self.rss_

        return self.A_


def pgd_fit_transform(X, A, B, W, ZWXt, ZXXt, *, max_iter, tol, batch_size, verbose, **params):
    return _pgd_like_optimize_aa(
        X,
        A,
        B,
        W,
        ZWXt,
        ZXXt,
        max_iter=max_iter,
        tol=tol,
        batch_size=batch_size,
        verbose=verbose,
        pseudo_pgd=False,
        update_B=True,
        **params,
    )


def pseudo_pgd_fit_transform(
    X, A, B, W, ZWXt, ZXXt, *, max_iter, tol, batch_size, verbose, **params
):
    return _pgd_like_optimize_aa(
        X,
        A,
        B,
        W,
        ZWXt,
        ZXXt,
        max_iter=max_iter,
        tol=tol,
        batch_size=batch_size,
        verbose=verbose,
        pseudo_pgd=True,
        update_B=True,
        **params,
    )


def _pgd_like_optimize_aa(
    X,
    A,
    B,
    W,
    ZWXt,  # precomputed kernel between W (data to fit) and X (used to compute the archetypes)
    ZXXt,  # precomputed kernel between X and X
    *,
    max_iter,
    tol,
    batch_size,  # batch size for B updates
    verbose=False,
    pseudo_pgd=False,
    update_B=True,
    step_size=1.0,
    max_iter_optimizer=10,
    beta=0.5,
    **params,
):

    batch_size = (
        X.shape[0] if batch_size is None else batch_size
    )  # if no batch size is given, use all data

    # precomputing and memory allocation
    # BX = archetypes

    BXWt = B @ W @ ZWXt.T
    BXXtBt = np.linalg.multi_dot([B, W, ZXXt, W.T, B.T])

    # Temporary arrays for updates
    A_grad = np.empty_like(A)
    A_new = np.empty_like(A)

    rss = -2 * np.sum(A.T * BXWt) + np.sum(BXXtBt.T * (A.T @ A))  # initial rss

    loss_list = [
        rss,
    ]

    step_size_A = step_size

    archetypes_idx_best = np.argmax(B, axis=1)
    n_archetypes = B.shape[0]

    # The algorithm alternates updates between A and B, in batches for B.

    # Allocate memory for B batch copies

    for i in range(1, max_iter + 1):  # main loop over iterations
        r = 0
        while r < B.shape[1]:
            # Define the batch indexes, ensuring archetypes are included
            r_max = min(r + batch_size, B.shape[1])
            idx = np.arange(r, r_max)

            # Ensure all archetypes are included in the batch, but keeping batch size
            missing_archetypes = np.setdiff1d(archetypes_idx_best, idx)
            n_missing = len(missing_archetypes)

            # Get idx without archetypes indexes
            idx_wo_archetypes = np.setdiff1d(idx, archetypes_idx_best)

            idx = idx_wo_archetypes[: (batch_size - n_archetypes)]
            idx = np.concatenate([idx, archetypes_idx_best])
            r += batch_size - n_missing

            idx_batch = np.arange(len(idx))

            # Define the batches
            B_batch = B[:, idx]
            ZWXt_batch = (ZWXt @ W)[:, idx]
            ZXXt_batch = (W @ ZXXt @ W.T)[np.ix_(idx, idx)]

            # Get the idx of archetypes in the batch
            archetypes_idx_best_batch = B_batch.argmax(axis=1)
            # Check all possible archetype updates in the batch
            for k in range(n_archetypes):
                archetypes_idx_batch = archetypes_idx_best_batch.copy()
                for i in idx_batch:
                    if i in archetypes_idx_best_batch:
                        continue
                    archetypes_idx_batch[k] = i

                    B_batch_new = np.zeros_like(B_batch)
                    B_batch_new[np.arange(B_batch.shape[0]), archetypes_idx_batch] = 1

                    BXWt_batch_new = B_batch_new @ ZWXt_batch.T
                    WXtBt_batch_new = ZWXt_batch @ B_batch_new.T
                    BXXtBt_batch_new = np.linalg.multi_dot([B_batch_new, ZXXt_batch, B_batch_new.T])

                    # Compute A update using new B_batch
                    A_copy = A.copy()
                    step_size_A_copy = step_size_A

                    rss_batch = np.inf
                    for _ in range(1, max_iter + 1):
                        rss_batch_new, step_size_A_copy = _pgd_like_update_A_inplace(
                            None,
                            A_copy,
                            ZWXt_batch,
                            ZXXt_batch,
                            BXWt_batch_new,
                            WXtBt_batch_new,
                            BXXtBt_batch_new,
                            A_grad,
                            A_new,  # used as temp storage, value is not important
                            pseudo_pgd,
                            step_size_A_copy,
                            max_iter_optimizer,
                            beta,
                            rss_batch,
                        )

                        convergence = abs(rss_batch_new - rss_batch) < tol
                        rss_batch = rss_batch_new
                        if convergence:
                            break

                    if rss_batch < rss:
                        rss = rss_batch
                        archetypes_idx_best_batch = archetypes_idx_batch.copy()
                        archetypes_idx_best = idx[archetypes_idx_best_batch]

                        B[:] = 0
                        B[np.arange(B.shape[0]), archetypes_idx_best] = 1

                        np.copyto(A, A_copy)

        convergence = abs(loss_list[-1] - rss) < tol
        loss_list.append(rss)
        if verbose and i % 10 == 0:
            verbose_print_rss(max_iter, rss, i)
        if convergence:
            break

        # Post processing, if weights are given
        archetypes_idx_best = np.argmax(B, axis=1)
        n_archetypes = B.shape[0]

        ZZt = (B @ W @ ZXXt @ W.T @ B.T).copy()

        rss_post = -2 * np.sum(BXXtBt * (ZZt)) + np.sum(BXXtBt * BXXtBt)  # initial rss

        for i in range(1, max_iter + 1):  # main loop over iterations
            r = 0
            while r < B.shape[1]:
                # Define the batch indexes, ensuring archetypes are included
                r_max = min(r + batch_size, B.shape[1])
                idx = np.arange(r, r_max)

                # Ensure all archetypes are included in the batch, but keeping batch size
                missing_archetypes = np.setdiff1d(archetypes_idx_best, idx)
                n_missing = len(missing_archetypes)

                # Get idx without archetypes indexes
                idx_wo_archetypes = np.setdiff1d(idx, archetypes_idx_best)

                idx = idx_wo_archetypes[: (batch_size - n_archetypes)]
                idx = np.concatenate([idx, archetypes_idx_best])
                r += batch_size - n_missing

                idx_batch = np.arange(len(idx))

                # Define the batches
                B_batch = B[:, idx]
                ZWXt_batch = (ZWXt)[:, idx]
                ZXXt_batch = (ZXXt)[np.ix_(idx, idx)]

                # Get the idx of archetypes in the batch
                archetypes_idx_best_batch = B_batch.argmax(axis=1)
                # Check all possible archetype updates in the batch
                for k in range(n_archetypes):
                    archetypes_idx_batch = archetypes_idx_best_batch.copy()
                    for i in idx_batch:
                        if i in archetypes_idx_best_batch:
                            continue
                        archetypes_idx_batch[k] = i

                        B_batch_new = np.zeros_like(B_batch)
                        B_batch_new[np.arange(B_batch.shape[0]), archetypes_idx_batch] = 1

                        BXWt_batch_new = B_batch_new @ ZWXt_batch.T
                        WXtBt_batch_new = ZWXt_batch @ B_batch_new.T
                        BXXtBt_batch_new = np.linalg.multi_dot(
                            [B_batch_new, ZXXt_batch, B_batch_new.T]
                        )

                        rss_batch = -2 * np.sum(BXXtBt_batch_new * (ZZt)) + np.sum(
                            BXXtBt_batch_new * BXXtBt_batch_new
                        )

                        if rss_batch < rss_post:
                            rss_post = rss_batch
                            archetypes_idx_best_batch = archetypes_idx_batch.copy()
                            archetypes_idx_best = idx[archetypes_idx_best_batch]

                            B[:] = 0
                            B[np.arange(B.shape[0]), archetypes_idx_best] = 1

                            np.copyto(A, A_copy)

            convergence = abs(loss_list[-1] - rss_post) < tol
            # loss_list.append(rss_post)
            if verbose and i % 10 == 0:
                verbose_print_rss(max_iter, rss_post, i)
            if convergence:
                break

        # Recompute A
        rss_batch = np.inf

        ZWXt_batch = ZWXt[:, archetypes_idx_best]
        ZXXt_batch = ZXXt[np.ix_(archetypes_idx_best, archetypes_idx_best)]
        BXWt_batch_new = B[:, archetypes_idx_best] @ ZWXt_batch.T
        WXtBt_batch_new = ZWXt_batch @ B[:, archetypes_idx_best].T
        BXXtBt_batch_new = np.linalg.multi_dot(
            [B[:, archetypes_idx_best], ZXXt_batch, B[:, archetypes_idx_best].T]
        )
        for _ in range(1, max_iter + 1):
            rss_batch_new, step_size_A = _pgd_like_update_A_inplace(
                None,
                A,
                ZWXt_batch,
                ZXXt_batch,
                BXWt_batch_new,
                WXtBt_batch_new,
                BXXtBt_batch_new,
                A_grad,
                A_new,  # used as temp storage, value is not important
                pseudo_pgd,
                step_size_A,
                max_iter_optimizer,
                beta,
                rss_batch,
            )

            convergence = abs(rss_batch_new - rss_batch) < tol
            rss_batch = rss_batch_new
            if convergence:
                break

    return A, B, None, i, loss_list, convergence


def _pgd_like_update_A_inplace(
    X,
    A,
    ZWXt,
    ZXXt,
    BXWt,
    WXtBt,
    BXXtBt,
    A_grad,
    A_new,
    pseudo_pgd,
    step_size_A,
    max_iter_optimizer,
    beta,
    rss,
):
    # gradient wrt A
    A_grad = np.matmul(A, BXXtBt, out=A_grad)
    A_grad -= WXtBt
    # A_grad /= (np.trace(XXt / A.shape[0]))
    # A_grad -= np.sum(A_grad * A, axis=1, keepdims=True)

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

        rss_new = -2 * np.sum(A_new.T * BXWt) + np.sum(BXXtBt.T * (A_new.T @ A_new))
        # print(f"RSS new: {rss_new}, RSS new 2: {rss_new_2}")

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


def verbose_print_rss(max_iter, rss, i):
    print(f"Iteration {i}/{max_iter}: RSS = {rss}")
