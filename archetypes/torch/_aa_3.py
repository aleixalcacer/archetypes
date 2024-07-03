from dataclasses import dataclass

import torch
from custom_inherit import doc_inherit

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
        device=None,
        generator=None,
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
            device=device,
            generator=generator,
        )

    def _init_A(self, X):
        A = torch.zeros((X.shape[0], self.n_archetypes), dtype=X.dtype, device=self.device)

        ind = torch.randint(0, self.n_archetypes, (X.shape[0],), generator=self.generator)

        for i, j in enumerate(ind):
            A[i, j] = 1

        return A

    def _init_B(self, X):
        B = torch.zeros((self.n_archetypes, X.shape[0]), dtype=X.dtype, device=self.device)

        ind = self.init_c_(X, self.n_archetypes, generator=self.generator, kwargs=self.init_kwargs)

        for i, j in enumerate(ind):
            B[i, j] = 1

        return B

    def _optim_A(self, X):
        pass

    def _optim_B(self, X):
        pass

    def _compute_archetypes(self, X):
        self.archetypes_ = self.B_ @ X

    def _loss(self, X):
        X_hat = self.A_ @ self.archetypes_
        return torch.linalg.norm(X - X_hat) ** 2

    def fit(self, X, y=None, **fit_params):
        # Initialize coefficients
        self.A_ = self._init_A(X)
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
        self.similarity_degree_ = self.A_.cpu().detach()
        self.archetypes_similarity_degree_ = self.B_.cpu().detach()
        self.labels_ = torch.argmax(self.A_, dim=1).cpu().detach()
        self.archetypes_ = self.archetypes_.cpu().detach()

        return self

    def transform(self, X):
        return self._optim_A(X).cpu().detach()

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


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
        device=None,
        generator=None,
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
            device=device,
            generator=generator,
        )

        self._check_parameters_()

    def _check_parameters_(self):
        # Check params for the optimization method
        if self.method == "autogd":
            self.method_c_: AAOptimizer = autogd_optimizer
            self.optimizer = self.method_kwargs.get("optimizer", "SGD")
            self.optimizer_kwargs = self.method_kwargs.get("optimizer_kwargs", {"lr": 1e-3})
            if not isinstance(self.optimizer, torch.optim.Optimizer):
                self.optimizer = getattr(torch.optim, self.optimizer)

            # TODO: Check if params are valid for the optimization method
        else:
            raise ValueError(f"Invalid method: {self.method}")

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


def _autogd_init_A(self, X):
    A_ = super(type(self), self)._init_A(X)
    self.A_opt_ = torch.asarray(A_, requires_grad=True)
    self.optimizer_A = self.optimizer(params=[self.A_opt_], **self.optimizer_kwargs)
    return A_


def _autogd_init_B(self, X):
    B_ = super(type(self), self)._init_B(X)
    self.B_opt_ = torch.asarray(B_, requires_grad=True)
    self.optimizer_B = self.optimizer(params=[self.B_opt_], **self.optimizer_kwargs)
    return B_


def _autogd_optim_A(self, X):
    loss = self._loss(X)
    self.optimizer_A.zero_grad()
    loss.backward(retain_graph=True)
    self.optimizer_A.step()
    return torch.softmax(self.A_opt_, dim=1)


def _autogd_optim_B(self, X):
    loss = self._loss(X)
    self.optimizer_B.zero_grad()
    loss.backward(retain_graph=True)
    self.optimizer_B.step()
    return torch.softmax(self.B_opt_, dim=1)


def _autogd_fit(self, X, y=None, **fit_params):
    return super(type(self), self).fit(X, y, **fit_params)


autogd_optimizer = AAOptimizer(
    A_init=_autogd_init_A,
    B_init=_autogd_init_B,
    A_optimize=_autogd_optim_A,
    B_optimize=_autogd_optim_B,
    fit=_autogd_fit,
)
