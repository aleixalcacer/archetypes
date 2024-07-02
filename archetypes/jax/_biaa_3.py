from dataclasses import dataclass

import jax
import optax
from custom_inherit import doc_inherit
from jax import grad
from jax import nn as jnn
from jax import numpy as jnp

from ..utils import arch_einsum
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
        seed=None,
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
            seed=seed,
        )

    def _init_A(self, X):
        # TODO: Improve it?
        A_0 = jnp.zeros((X.shape[0], self.n_archetypes[0]), dtype=jnp.float32)

        self.key, subkey = jax.random.split(self.key)
        ind = jax.random.choice(subkey, self.n_archetypes[0], (X.shape[0],), replace=True)

        for i, j in enumerate(ind):
            A_0 = A_0.at[i, j].set(1.0)

        A_1 = jnp.zeros((X.shape[1], self.n_archetypes[1]), dtype=jnp.float32)

        self.key, subkey = jax.random.split(self.key)
        ind = jax.random.choice(subkey, self.n_archetypes[1], (X.shape[1],), replace=True)

        for i, j in enumerate(ind):
            A_1 = A_1.at[i, j].set(1.0)

        return [A_0, A_1]

    def _init_B(self, X):
        B_0 = jnp.zeros((self.n_archetypes[0], X.shape[0]), dtype=jnp.float32)

        ind, self.key = self.init_c_(X, self.n_archetypes[0], key=self.key, kwargs=self.init_kwargs)

        for i, j in enumerate(ind):
            B_0 = B_0.at[i, j].set(1.0)

        B_1 = jnp.zeros((self.n_archetypes[1], X.shape[1]), dtype=jnp.float32)

        ind, self.key = self.init_c_(
            X.T, self.n_archetypes[1], key=self.key, kwargs=self.init_kwargs
        )

        for i, j in enumerate(ind):
            B_1 = B_1.at[i, j].set(1.0)

        return [B_0, B_1]

    def _optim_A(self, X):
        pass

    def _optim_B(self, X):
        pass

    def _compute_archetypes(self, X):
        self.archetypes_ = arch_einsum(self.B_, X)

    def _loss(self, X):
        X_hat = arch_einsum(self.A_, self.archetypes_)
        return jnp.linalg.norm(X - X_hat) ** 2

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
        self.labels_ = [jnp.argmax(A_i, axis=1) for A_i in self.A_]

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
        seed=None,
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
            seed=seed,
        )

        # Check params for the optimization method
        if self.method == "autogd":
            self.method_c_: BiAAOptimizer = jax_biaa_optimizer
            self.optimizer = self.method_kwargs.get("optimizer", "sgd")
            self.optimizer_kwargs = self.method_kwargs.get(
                "optimizer_kwargs", {"learning_rate": 1e-3}
            )
            if not isinstance(self.optimizer, optax.GradientTransformation):
                self.optimizer = getattr(optax, self.optimizer)

            # TODO: Check if params are valid for the optimization method
        else:
            raise ValueError(f"Method {self.method} is not supported in Jax backend.")

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


# Gradient Descent (JAX)
def _jax_biaa_init_A(self, X):
    self.A_ = super(type(self), self)._init_A(X)
    self.A_opt_ = tuple([jnp.asarray(A_i, copy=True) for A_i in self.A_])

    self.params_A = (self.A_opt_[0], self.A_opt_[1])

    self.optimizer_A_state = self.optimizer_A.init(self.params_A)

    return self.A_opt_


def _jax_bia_init_B(self, X):
    self.B_ = super(type(self), self)._init_B(X)
    self.B_opt_ = tuple([jnp.asarray(B_i, copy=True) for B_i in self.B_])

    self.params_B = (self.B_[0], self.B_[1])

    self.optimizer_B_state = self.optimizer_B.init(self.params_B)

    return self.B_opt_


def jax_biaa_optim_A(self, X):
    grad_A = grad(_jax_biaa_loss, argnums=[0, 1])(*self.A_, *self.B_, X)
    updates_A, self.optimizer_A_state = self.optimizer_A.update(grad_A, self.optimizer_A_state)
    self.params_A = optax.apply_updates(self.params_A, updates_A)

    self.A_opt_ = [self.params_A[0], self.params_A[1]]

    return [jnn.softmax(A_i, axis=1) for A_i in self.A_opt_]


def jax_biaa_optim_B(self, X):
    grad_B = grad(_jax_biaa_loss, argnums=[2, 3])(*self.A_, *self.B_, X)
    updates_B, self.optimizer_B_state = self.optimizer_B.update(grad_B, self.optimizer_B_state)
    self.params_B = optax.apply_updates(self.params_B, updates_B)

    self.B_opt_ = [self.params_B[0], self.params_B[1]]

    return [jnn.softmax(B_i, axis=1) for B_i in self.B_opt_]


def _jax_biaa_fit(self, X, y=None, **fit_params):
    # Pre-computations for optimization
    self.optimizer_A = self.optimizer(**self.optimizer_kwargs)
    self.optimizer_B = self.optimizer(**self.optimizer_kwargs)
    return super(type(self), self).fit(X, y, **fit_params)


def _jax_biaa_loss(A_0, A_1, B_0, B_1, X):
    X1 = X
    Z = B_0 @ X @ B_1.T
    X2 = A_0 @ Z @ A_1.T

    return optax.l2_loss(X1 - X2).sum()


# TODO: Check advanced usage of optax to improve the optimizer
jax_biaa_optimizer = BiAAOptimizer(
    A_init=_jax_biaa_init_A,
    B_init=_jax_bia_init_B,
    A_optimize=jax_biaa_optim_A,
    B_optimize=jax_biaa_optim_B,
    fit=_jax_biaa_fit,
)
