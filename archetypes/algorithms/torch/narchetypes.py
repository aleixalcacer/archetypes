import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .utils import einsum, einsum_dc, loss_fun, softmax


def map_params(params, relations):
    return [params[r] for r in relations]


class NAA(nn.Module):
    """
    N-Archetype analysis implemented in PyTorch.

    Parameters
    ----------
    k: tuple
        The number of archetypes to use for each dimension.

    s: tuple
        The number of observations in each dimension.

    device: str
        The device to use for training the model. Defaults to "cpu".
    """

    def __init__(
        self,
        n_archetypes,
        shape,
        relations=None,
        degree_correction=False,
        membership="soft",
        loss="normal",
        device="cpu",
    ):
        super().__init__()

        if len(n_archetypes) > len(shape):
            raise ValueError(
                "The number of archetypes must be less or equal than the number of dimensions."
            )

        self.k = n_archetypes
        self.s = shape
        # compute difference between k and s dimensions
        self.n_free_dim = len(self.s) - len(self.k)

        self.n = len(n_archetypes)
        self.device = device

        if membership not in ["soft"]:
            raise ValueError("membership must be one of 'soft'")
        self.membership = membership

        if loss not in ["normal", "bernoulli", "poisson"]:
            raise ValueError("loss must be one of 'normal', 'bernoulli', 'poisson'")
        self.loss = loss

        # relations
        if relations is None:
            relations = list(np.arange(self.n))
        self.relations = relations

        relations_s = dict(zip(self.relations, self.s))
        relations_k = dict(zip(self.relations, self.k))

        # unique sorted relations
        relations_unique = sorted(set(relations))

        self._DC = None
        if degree_correction:
            self._DC_params = [
                torch.nn.Parameter(
                    torch.randn(relations_s[r], device=self.device), requires_grad=True
                )
                for r in relations_unique
            ]
            self._DC = map_params(self._DC_params, relations)

        # data-membership matrices
        self._A_params = [
            torch.nn.Parameter(
                torch.randn(relations_s[r], relations_k[r], device=self.device), requires_grad=True
            )
            for r in relations_unique
        ]

        self._A = map_params(self._A_params, relations)

        # archetype-membership matrices
        self._B_params = [
            torch.nn.Parameter(
                torch.randn(relations_k[r], relations_s[r], device=self.device), requires_grad=True
            )
            for r in relations_unique
        ]

        self._B = map_params(self._B_params, relations)

        # archetypes
        self._Z = None

        self.losses = []

    def _loss(self):
        """
        The negative log-likelihood of a normal distribution
        """

        X1 = self._X
        Z = einsum(self.B, X1)
        X2 = einsum(self.A, Z)
        if self._DC:
            X2 = einsum_dc(self.DC, X2)

        return loss_fun(X1, X2, self.loss).sum()

    def fit(self, data, n_epochs, learning_rate=0.01):
        """
        Fit the model.

        Parameters
        ----------

        data: torch.Tensor
            The data to be used for training.

        n_epochs: int
            The number of epochs to train the model for.

        learning_rate: float
            The learning rate to use for training.
        """

        self._X = data.to(self.device)
        self._Z = einsum(self.B, self._X)

        params = [*self._A_params, *self._B_params]
        if self._DC:
            params += self._DC_params

        optimizer = torch.optim.Adam(params=params, lr=learning_rate)

        pbar_epoch = tqdm(range(n_epochs), leave=True)

        for i, epoch in enumerate(pbar_epoch):
            loss = self._loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            self.losses.append(loss_item)
            pbar_epoch.set_description(f"Epoch {epoch}/{n_epochs} | loss {loss_item:.4f}")

        self._Z = einsum(self.B, self._X)

    @property
    def DC(self):
        """
        The degree correction matrices.

        Returns
        -------
        list of torch.Tensor
        """
        if self._DC is None:
            return None
        return [torch.sigmoid(dc_i) for dc_i in self._DC]

    @property
    def A(self):
        """
        A coefficient matrices.

        Returns
        -------
        list of torch.Tensor
        """
        return [softmax(a_i, dim=1) for a_i in self._A]

    @property
    def B(self):
        """
        B coefficient matrices.

        Returns
        -------
        list of torch.Tensor
        """
        return [torch.softmax(b_i, dim=1) for b_i in self._B]

    @property
    def Z(self):
        """
        The archetype matrix.

        Returns
        -------
        torch.Tensor
        """
        return self._Z

    @property
    def estimated_data(self):
        """
        The estimated data matrix.

        Returns
        -------
        torch.Tensor
        """
        data = einsum(self.A, self.Z)
        if self._DC:
            data = einsum_dc(self.DC, data)
        return data
