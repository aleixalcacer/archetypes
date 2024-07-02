from dataclasses import dataclass

import jax
import optax
from custom_inherit import doc_inherit
from jax import grad
from jax import nn as jnn
from jax import numpy as jnp

from ._base import AABase


@dataclass
class AAOptimizer:
    A_init: callable
    B_init: callable
    A_optimize: callable
    B_optimize: callable
    fit: callable


@doc_inherit(parent=AABase, style="numpy_with_merge")
class AABase_3(AABase):
    """
    Archetype Analysis.
    """

    def __init__(
        self,
        n_archetypes,
        max_iter=300,
        tol=1e-4,
        init="uniform",
        init_kwargs=None,
        save_init=False,
        verbose=False,
        seed=None,
        method="autogd",
        method_kwargs=None,
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
        A = jnp.zeros((X.shape[0], self.n_archetypes), dtype=jnp.float64)

        self.key, subkey = jax.random.split(self.key)
        ind = jax.random.choice(subkey, self.n_archetypes, (X.shape[0],), replace=True)

        for i, j in enumerate(ind):
            A.at[i, j].set(1)

        return A

    def _init_B(self, X):
        B = jnp.zeros((self.n_archetypes, X.shape[0]), dtype=jnp.float64)

        ind, self.key = self.init_c_(X, self.n_archetypes, key=self.key, kwargs=self.init_kwargs)

        for i, j in enumerate(ind):
            B.at[i, j].set(1)

        return B

    def _optim_A(self, X):
        pass

    def _optim_B(self, X):
        pass

    def _compute_archetypes(self, X):
        self.archetypes_ = self.B_ @ X

    def _loss(self, X):
        X_hat = self.A_ @ self.archetypes_
        return jnp.linalg.norm(X - X_hat) ** 2

    def fit(self, X, y=None, **fit_params):
        # Initialize coefficients
        self.A_ = self._init_A(X)
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
        self.labels_ = jnp.argmax(self.A_, axis=1)

        return self

    def transform(self, X):
        return self._optim_A(X)


@doc_inherit(parent=AABase_3, style="numpy_with_merge")
class AA_3(AABase_3):
    """
    Archetype Analysis s.t. |X - ABX|_2^2 is minimized.
    """

    def __init__(
        self,
        n_archetypes,
        max_iter=300,
        tol=1e-4,
        init="uniform",
        init_kwargs=None,
        save_init=False,
        verbose=False,
        seed=None,
        method="nnls",
        method_kwargs=None,
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

        self._check_parameters_()

    def _check_parameters_(self):
        # Check params for the optimization method
        if self.method == "autogd":
            self.method_c_: AAOptimizer = jax_optimizer
            self.optimizer = self.method_kwargs.get("optimizer", "sgd")
            self.optimizer_kwargs = self.method_kwargs.get(
                "optimizer_kwargs", {"learning_rate": 1e-3}
            )
            if not isinstance(self.optimizer, optax.GradientTransformation):
                self.optimizer = getattr(optax, self.optimizer)

            # TODO: Check if params are valid for the optimization method
        else:
            raise ValueError(f"Method {self.method} is not supported in Jax backend.")

    def _init_B(self, X):
        return self.method_c_.B_init(self, X)

    def _init_A(self, X):
        return self.method_c_.A_init(self, X)

    def _optim_A(self, X):
        return self.method_c_.A_optimize(self, X)

    def _optim_B(self, X):
        return self.method_c_.B_optimize(self, X)

    def fit(self, X, y=None, **fit_params):
        return self.method_c_.fit(self, X, y, **fit_params)


# Gradient Descent (JAX)
def _jax_init_A(self, X):
    self.A_ = super(type(self), self)._init_A(X)
    self.A_opt_ = jnp.asarray(self.A_, copy=True)
    self.optimizer_A_state = self.optimizer_A.init(self.A_opt_)
    return self.A_opt_


def _jax_init_B(self, X):
    self.B_ = super(type(self), self)._init_B(X)
    self.B_opt_ = jnp.asarray(self.B_, copy=True)
    self.optimizer_B_state = self.optimizer_B.init(self.B_opt_)
    return self.B_opt_


def jax_optim_A(self, X):
    grad_A = grad(_jax_loss, argnums=1)(X, self.A_, self.B_)
    updates_A, self.optimizer_A_state = self.optimizer_A.update(grad_A, self.optimizer_A_state)
    self.A_opt_ = optax.apply_updates(self.A_opt_, updates_A)
    return jnn.softmax(self.A_opt_, axis=1)


def jax_optim_B(self, X):
    grad_B = grad(_jax_loss, argnums=2)(X, self.A_, self.B_)
    updates_B, self.optimizer_B_state = self.optimizer_B.update(grad_B, self.optimizer_B_state)
    self.B_opt_ = optax.apply_updates(self.B_opt_, updates_B)
    return jnn.softmax(self.B_opt_, axis=1)


def _jax_fit(self, X, y=None, **fit_params):
    # Pre-computations for optimization
    self.optimizer_A = self.optimizer(**self.optimizer_kwargs)
    self.optimizer_B = self.optimizer(**self.optimizer_kwargs)

    return super(type(self), self).fit(X, y, **fit_params)


def _jax_loss(X, A, B):
    return optax.l2_loss(X - A @ B @ X).sum()


jax_optimizer = AAOptimizer(
    A_init=_jax_init_A,
    B_init=_jax_init_B,
    A_optimize=jax_optim_A,
    B_optimize=jax_optim_B,
    fit=_jax_fit,
)
