import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from .utils import einsum


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

    def __init__(self, k, s, device="cpu"):
        super().__init__()

        # Check that k and s has the same length
        assert len(k) == len(s), "k and s must have the same length"

        self.k = k
        self.s = s
        self.n = len(k)
        self.device = device

        self._A = [
            torch.nn.Parameter(torch.randn(s_i, k_i, device=self.device), requires_grad=True)
            for s_i, k_i in zip(self.s, self.k)
        ]

        self._B = [
            torch.nn.Parameter(torch.randn(k_i, s_i, device=self.device), requires_grad=True)
            for s_i, k_i in zip(self.s, self.k)
        ]

        self._Z = None

        self.losses = []

    def _loss(self):
        """
        The negative log-likelihood of a normal distribution
        """

        X1 = self._X
        Z = einsum(self.B, X1)
        X2 = einsum(self.A, Z)

        loss = torch.pow(X1 - X2, 2).sum()

        return loss

    def train(self, data, n_epochs, learning_rate=0.01):
        """
        Train the model.

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

        optimizer = torch.optim.Adam(params=[*self._A, *self._B], lr=learning_rate)

        pbar_epoch = tqdm(range(n_epochs), leave=True)

        for i, epoch in enumerate(pbar_epoch):
            loss = self._loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.losses.append(loss.item())
            pbar_epoch.set_description(f"Epoch {epoch}/{n_epochs} | loss {loss.item():.4f}")

        self._Z = einsum(self.B, self._X)
        plt.close()

    @property
    def A(self):
        """
        A coefficient matrices.

        Returns
        -------
        list of torch.Tensor
        """
        return [torch.softmax(a_i, dim=1) for a_i in self._A]

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
        The archetypes matrix.

        Returns
        -------
        torch.Tensor
        """
        return self._Z
