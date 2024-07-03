from dataclasses import dataclass

import torch
from custom_inherit import doc_inherit

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
        device=None,
        generator=None,
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
            device=device,
            generator=generator,
        )

    def _init_A(self, X):
        # TODO: Improve it?
        A_0 = torch.zeros((X.shape[0], self.n_archetypes[0]), dtype=X.dtype)

        ind = torch.randint(0, self.n_archetypes[0], (X.shape[0],), generator=self.generator)

        for i, j in enumerate(ind):
            A_0[i, j] = 1

        A_1 = torch.zeros((X.shape[1], self.n_archetypes[1]), dtype=X.dtype)

        ind = torch.randint(0, self.n_archetypes[1], (X.shape[1],), generator=self.generator)

        for i, j in enumerate(ind):
            A_1[i, j] = 1

        return [A_0, A_1]

    def _init_B(self, X):
        B_0 = torch.zeros((self.n_archetypes[0], X.shape[0]), dtype=X.dtype)

        ind = self.init_c_(
            X, self.n_archetypes[0], generator=self.generator, kwargs=self.init_kwargs
        )

        for i, j in enumerate(ind):
            B_0[i, j] = 1

        B_1 = torch.zeros((self.n_archetypes[1], X.shape[1]), dtype=X.dtype)

        ind = self.init_c_(
            X.T, self.n_archetypes[1], generator=self.generator, kwargs=self.init_kwargs
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
        return torch.linalg.norm(X - X_hat) ** 2

    def fit(self, X, y=None, **fit_params):
        # Initialize coefficients
        self.A_ = self._init_A(X)  # Initialize A uniformly
        self.B_ = self._init_B(X)

        self._compute_archetypes(X)

        if self.save_init:
            self.archetypes_init_ = self.archetypes_.copy()

        rss = self._loss(X).item()
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
            rss = self._loss(X).item()
            self.loss_.append(rss)
            if abs(self.loss_[-1] - self.loss_[-2]) < self.tol:
                break

        # Set attributes
        self.similarity_degree_ = [a.cpu().detach() for a in self.A_]
        self.archetypes_similarity_degree_ = [b.cpu().detach() for b in self.B_]
        self.labels_ = [torch.argmax(A_i, dim=1).cpu().detach() for A_i in self.A_]
        self.archetypes_ = self.archetypes_.cpu().detach()

        return self

    def transform(self, X):
        A_ = self._optim_A(X)
        return [a.cpu().detach() for a in A_]

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
        verbose=False,
        device=None,
        generator=None,
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
            device=device,
            generator=generator,
        )

        # Check params for the optimization method
        if self.method == "autogd":
            self.method_c_: BiAAOptimizer = autogd_biaa_optimizer
            self.optimizer = self.method_kwargs.get("optimizer", "SGD")
            self.optimizer_kwargs = self.method_kwargs.get("optimizer_kwargs", {"lr": 1e-3})
            if not isinstance(self.optimizer, torch.optim.Optimizer):
                self.optimizer = getattr(torch.optim, self.optimizer)

            # TODO: Check if params are valid for the optimization method
        else:
            raise ValueError(f"Invalid method: {self.method}")

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


# Autograd optimizer
def _autogd_biaa_init_A(self, X):
    A_ = super(type(self), self)._init_A(X)
    self.A_opt_ = [torch.asarray(a, requires_grad=True) for a in A_]
    self.optimizer_A = self.optimizer(params=[*self.A_opt_], **self.optimizer_kwargs)
    return A_


def _autogd_biaa_init_B(self, X):
    B_ = super(type(self), self)._init_B(X)
    self.B_opt_ = [torch.asarray(b, requires_grad=True) for b in B_]
    self.optimizer_B = self.optimizer(params=[*self.B_opt_], **self.optimizer_kwargs)
    return B_


def _autogd_biaa_optim_A(self, X):
    loss = self._loss(X)
    self.optimizer_A.zero_grad()
    loss.backward(retain_graph=True)
    self.optimizer_A.step()
    return [torch.softmax(a, dim=1) for a in self.A_opt_]


def _autogd_biaa_optim_B(self, X):
    loss = self._loss(X)
    self.optimizer_B.zero_grad()
    loss.backward(retain_graph=True)
    self.optimizer_B.step()
    return [torch.softmax(b, dim=1) for b in self.B_opt_]


def _autogd_biaa_fit(self, X, y=None, **fit_params):
    return super(type(self), self).fit(X, y, **fit_params)


autogd_biaa_optimizer = BiAAOptimizer(
    A_init=_autogd_biaa_init_A,
    B_init=_autogd_biaa_init_B,
    A_optimize=_autogd_biaa_optim_A,
    B_optimize=_autogd_biaa_optim_B,
    fit=_autogd_biaa_fit,
)
