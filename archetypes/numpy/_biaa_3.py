from dataclasses import dataclass

import numpy as np
from custom_inherit import doc_inherit

from ..utils import arch_einsum, einsum, nnls
from ._base import BiAABase


@dataclass
class BiAAOptimizer:
    A_init: callable
    B_init: callable
    A_optimize: callable
    B_optimize: callable
    fit: callable


@doc_inherit(parent=BiAABase, style="numpy_with_merge")
class BiAABase_3(BiAABase):
    """
    Base class for factorizing a data matrix into three matrices s.t.
    F = | X - A_0B_0XB_1A_1| is minimal.
    """

    def __init__(
        self,
        n_archetypes,
        max_iter=300,
        tol=1e-4,
        init="uniform",
        init_kwargs=None,
        save_init=False,
        method="nnls",
        method_kwargs=None,
        verbose=False,
        random_state=None,
    ):
        super().__init__(
            n_archetypes=n_archetypes,
            max_iter=max_iter,
            tol=tol,
            init=init,
            init_kwargs=init_kwargs,
            save_init=save_init,
            method=method,
            method_kwargs=method_kwargs,
            verbose=verbose,
            random_state=random_state,
        )

    def _init_A(self, X):
        # TODO: Improve it?
        A_0 = np.zeros((X.shape[0], self.n_archetypes[0]), dtype=np.float64)

        ind = self.random_state.choice(self.n_archetypes[0], X.shape[0], replace=True)

        for i, j in enumerate(ind):
            A_0[i, j] = 1

        A_1 = np.zeros((X.shape[1], self.n_archetypes[1]), dtype=np.float64)

        ind = self.random_state.choice(self.n_archetypes[1], X.shape[1], replace=True)

        for i, j in enumerate(ind):
            A_1[i, j] = 1

        return [A_0, A_1]

    def _init_B(self, X):
        B_0 = np.zeros((self.n_archetypes[0], X.shape[0]), dtype=np.float64)

        ind = self.init_c_(
            X, self.n_archetypes[0], random_state=self.random_state, kwargs=self.init_kwargs
        )

        for i, j in enumerate(ind):
            B_0[i, j] = 1

        B_1 = np.zeros((self.n_archetypes[1], X.shape[1]), dtype=np.float64)

        ind = self.init_c_(
            X.T, self.n_archetypes[1], random_state=self.random_state, kwargs=self.init_kwargs
        )

        for i, j in enumerate(ind):
            B_1[i, j] = 1

        return [B_0, B_1]

    def _optim_A(self, X):
        pass

    def _optim_B(self, X):
        pass

    def _compute_archetypes(self, X):
        self.archetypes_ = arch_einsum(self.B_, X)

    def _loss(self, X):
        X_hat = arch_einsum(self.A_, self.archetypes_)
        return np.linalg.norm(X - X_hat) ** 2

    def fit(self, X, y=None, **fit_params):
        # Initialize coefficients
        self.A_ = self._init_A(X)  # Initialize A uniformly
        self.B_ = self._init_B(X)

        self._compute_archetypes(X)

        if self.save_init:
            self.archetypes_init_ = self.archetypes_.copy()

        rss = self._loss(X)
        self.loss_ = [rss]

        for i in range(self.max_iter):
            # Verbose mode (print RSS)
            if self.verbose and i % 10 == 0:
                print(f"Iteration {i}/{self.max_iter}: RSS = {rss}")

            # Optimize coefficients
            self.A_ = self._optim_A(X)
            self.B_ = self._optim_B(X)

            self._compute_archetypes(X)

            # Compute RSS
            rss = self._loss(X)
            self.loss_.append(rss)
            if abs(self.loss_[-1] - self.loss_[-2]) < self.tol:
                break

        # Set attributes
        self.similarity_degree_ = self.A_
        self.archetypes_similarity_degree_ = self.B_
        self.labels_ = [np.argmax(A_i, axis=1) for A_i in self.A_]

        return self

    def transform(self, X):
        return self._optim_A(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


@doc_inherit(parent=BiAABase_3, style="numpy_with_merge")
class BiAA_3(BiAABase_3):
    """
    BiArchetype Analysis s.t. F = | X - A_0B_0XB_1A_1| is minimal.
    """

    def __init__(
        self,
        n_archetypes,
        max_iter=300,
        tol=1e-4,
        init="uniform",
        init_kwargs=None,
        save_init=False,
        method="nnls",
        method_kwargs=None,
        verbose=False,
        random_state=None,
    ):
        super().__init__(
            n_archetypes=n_archetypes,
            max_iter=max_iter,
            tol=tol,
            init=init,
            init_kwargs=init_kwargs,
            save_init=save_init,
            method=method,
            method_kwargs=method_kwargs,
            verbose=verbose,
            random_state=random_state,
        )

        # Check params for the optimization method
        if self.method == "nnls":
            self.method_c_: BiAAOptimizer = nnls_biaa_optimizer  # dataclass
            self.max_iter_optimizer = self.method_kwargs.get("max_iter_optimizer", 100)
            self.const = self.method_kwargs.get("const", 100.0)

        # TODO: Check if the parameters are valid for the optimization method

    def _init_A(self, X):
        return self.method_c_.A_init(self, X)

    def _init_B(self, X):
        return self.method_c_.B_init(self, X)

    def _optim_B(self, X):
        return self.method_c_.B_optimize(self, X)

    def _optim_A(self, X):
        return self.method_c_.A_optimize(self, X)

    def fit(self, X, y=None, **fit_params):
        return self.method_c_.fit(self, X, y, **fit_params)


# Non-Negative Least Squares
def _nnls_biaa_init_A(self, X):
    return super(type(self), self)._init_A(X)


def _nnls_biaa_init_B(self, X):
    return super(type(self), self)._init_B(X)


def _nnls_biaa_optim_B(self, X):
    B_ = np.linalg.pinv(self.A_[0]) @ X
    X_ = einsum([X, self.B_[1].T, self.A_[1].T])
    B_0 = nnls(B_, X_, max_iter_optimizer=self.max_iter_optimizer, const=self.const)

    B_ = (X @ np.linalg.pinv(self.A_[1].T)).T
    X_ = einsum([self.A_[0], B_0, X]).T
    B_1 = nnls(B_, X_, max_iter_optimizer=self.max_iter_optimizer, const=self.const)

    return [B_0, B_1]


def _nnls_biaa_optim_A(self, X):
    B_ = X
    X_ = einsum([self.B_[0], X, self.B_[1].T, self.A_[1].T])

    A_0 = nnls(B_, X_, max_iter_optimizer=self.max_iter_optimizer, const=self.const)

    B_ = X.T
    X_ = einsum([A_0, self.B_[0], X, self.B_[1].T]).T

    A_1 = nnls(B_, X_, max_iter_optimizer=self.max_iter_optimizer, const=self.const)

    return [A_0, A_1]


def _nnls_biaa_fit(self, X, y=None, **fit_params):
    return super(type(self), self).fit(X, y, **fit_params)


nnls_biaa_optimizer = BiAAOptimizer(
    A_init=_nnls_biaa_init_A,
    B_init=_nnls_biaa_init_B,
    A_optimize=_nnls_biaa_optim_A,
    B_optimize=_nnls_biaa_optim_B,
    fit=_nnls_biaa_fit,
)
