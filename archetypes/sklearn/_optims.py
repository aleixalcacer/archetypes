from dataclasses import dataclass

import numpy as np
import optax
from jax import grad
from jax import nn as jnn
from jax import numpy as jnp

from archetypes.utils import nnls


@dataclass
class AAOptimizer:
    A_init: callable
    B_init: callable
    A_optimize: callable
    B_optimize: callable
    fit: callable


# Non-Negative Least Squares


def _nnls_init_A(self, X):
    super(type(self), self)._init_A(X)


def _nnls_init_B(self, X):
    super(type(self), self)._init_B(X)


def _nnls_optim_B(self, X):
    self.archetypes_ = np.linalg.pinv(self.A_) @ X
    B_ = self.archetypes_
    X_ = X
    self.B_ = nnls(B_, X_, max_iter=self.max_iter_optimizer)


def _nnls_optim_A(self, X):
    B_ = X
    X_ = self.archetypes_
    self.A_ = nnls(B_, X_, max_iter=self.max_iter_optimizer)


def _nnls_fit(self, X, y=None, **fit_params):
    return super(type(self), self).fit(X, y, **fit_params)


nnls_optimizer = AAOptimizer(
    A_init=_nnls_init_A,
    B_init=_nnls_init_B,
    A_optimize=_nnls_optim_A,
    B_optimize=_nnls_optim_B,
    fit=_nnls_fit,
)

# Projected Gradient Descent (Closed Form)


def _pgd_init_A(self, X):
    super(type(self), self)._init_A(X)


def _pgd_init_B(self, X):
    super(type(self), self)._init_B(X)


def _pgd_optim_A(self, X):
    C_ = self.B_.T
    S_ = self.A_.T
    GS_ = self.GA_.T

    # Pre-computations
    CTXTXC = C_.T @ self.XTX_ @ C_
    CTXTX = C_.T @ self.XTX_
    SST = S_ @ S_.T
    rss_prev = -2 * np.sum(CTXTX * S_) + np.sum(CTXTXC * SST)

    for _ in range(self.n_iter_optimizer):
        # Compute the gradient
        GS_ = CTXTXC @ S_ - CTXTX
        GS_ = GS_ - np.sum(GS_ * S_, axis=0)

        # Optimize the step size
        S_prev = S_.copy()
        for _ in range(self.max_iter_optimizer):
            S_ = S_prev - self.step_size_A_ * GS_
            S_ = np.where(S_ < 0, 1e-8, S_)
            S_ = S_ / np.sum(S_, axis=0)

            SST = S_ @ S_.T
            rss = -2 * np.sum(CTXTX * S_) + np.sum(CTXTXC * SST)

            if rss <= rss_prev:
                self.step_size_A_ /= self.beta_
                rss_prev = rss
                break

            self.step_size_A_ *= self.beta_

    self.GA_ = GS_.T
    self.A_ = S_.T


def _pgd_optim_B(self, X):
    C_ = self.B_.T
    S_ = self.A_.T
    GC_ = self.GB_.T

    # Pre-computations
    SST = S_ @ S_.T
    XTXST = self.XTX_ @ S_.T

    rss_prev = -2 * np.sum(XTXST * C_) + np.sum(C_.T @ self.XTX_ @ C_ * SST)

    for _ in range(self.n_iter_optimizer):
        # Compute the gradient
        GC_ = self.XTX_ @ C_ @ SST - XTXST
        GC_ = GC_ - np.sum(GC_ * C_, axis=0)

        # Optimize the step size
        C_prev = C_.copy()
        for _ in range(self.max_iter_optimizer):
            C_ = C_prev - self.step_size_B_ * GC_
            C_ = np.where(C_ < 1e-8, 0, C_)
            C_ = C_ / np.sum(C_, axis=0)

            rss = -2 * np.sum(XTXST * C_) + np.sum(C_.T @ self.XTX_ @ C_ * SST)
            if rss <= rss_prev:
                self.step_size_B_ /= self.beta_
                rss_prev = rss
                break

            self.step_size_B_ *= self.beta_

    self.GB_ = GC_.T
    self.B_ = C_.T


def _pgd_fit(self, X, y=None, **fit_params):
    # Pre-computations for optimization
    self.GA_ = np.zeros((X.shape[0], self.n_archetypes))
    self.GB_ = np.zeros((self.n_archetypes, X.shape[0]))
    self.XTX_ = X @ X.T

    return super(type(self), self).fit(X, y, **fit_params)


pgd_optimizer = AAOptimizer(
    A_init=_pgd_init_A,
    B_init=_pgd_init_B,
    A_optimize=_pgd_optim_A,
    B_optimize=_pgd_optim_B,
    fit=_pgd_fit,
)


# Gradient Descent (JAX)


def _jax_init_A(self, X):
    super(type(self), self)._init_A(X)
    self.A_opt_ = jnp.asarray(self.A_, copy=True)
    self.A_ = self.A_opt_
    self.optimizer_A_state = self.optimizer_A.init(self.A_opt_)


def _jax_init_B(self, X):
    super(type(self), self)._init_B(X)
    self.B_opt_ = jnp.asarray(self.B_, copy=True)
    self.B_ = self.B_opt_
    self.optimizer_B_state = self.optimizer_B.init(self.B_opt_)


def jax_optim_A(self, X):
    grad_a = grad(_jax_loss, argnums=1)(X, self.A_, self.B_)
    updates, self.optimizer_A_state = self.optimizer_A.update(grad_a, self.optimizer_A_state)
    self.A_opt_ = optax.apply_updates(self.A_opt_, updates)
    self.A_ = jnn.softmax(self.A_opt_, axis=1)


def jax_optim_B(self, X):
    grad_b = grad(_jax_loss, argnums=2)(X, self.A_, self.B_)
    updates, self.optimizer_B_state = self.optimizer_B.update(grad_b, self.optimizer_B_state)
    self.B_opt_ = optax.apply_updates(self.B_opt_, updates)
    self.B_ = jnn.softmax(self.B_opt_, axis=1)


def _jax_fit(self, X, y=None, **fit_params):
    # Pre-computations for optimization
    self.optimizer_A = self.optimizer_c(**self.optimizer_kwargs)
    self.optimizer_B = self.optimizer_c(**self.optimizer_kwargs)

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
