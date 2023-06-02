import torch
import torch.nn as nn
from tqdm import tqdm


class AA(nn.Module):
    """
    Archetype analysis implemented in PyTorch.

    Parameters
    ----------
    k: int
        The number of archetypes to use.

    m: int
        The number of observations.

    n: int
        The number of variables.

    device: str
        The device to use for training the model. Defaults to "cpu".
    """

    def __init__(self, k, m, n, device="cpu"):
        super().__init__()

        self.m = m
        self.n = n
        self.k = k
        self.device = device

        self._A = torch.nn.Parameter(
            torch.randn(self.m, self.k, device=self.device), requires_grad=True
        )

        self._B = torch.nn.Parameter(
            torch.randn(self.k, self.m, device=self.device), requires_grad=True
        )

        self._Z = torch.zeros(self.k, self.n, device=self.device)

        self.losses = []

    def _loss(self):
        """
        The negative log-likelihood of a normal distribution
        """
        loss = torch.pow(self._X - self.A @ self.B @ self._X, 2).sum()
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

        optimizer = torch.optim.Adam(params=[self._A, self._B], lr=learning_rate)

        pbar_epoch = tqdm(range(n_epochs), leave=True)

        for epoch in pbar_epoch:
            loss = self._loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.losses.append(loss.item())
            pbar_epoch.set_description(f"Epoch {epoch}/{n_epochs} | loss {loss.item():.4f}")

        self._Z = self.B @ self._X

    @property
    def A(self):
        """
        A coefficient matrix.

        Returns
        -------
        torch.Tensor
        """
        return torch.softmax(self._A, dim=1)

    @property
    def B(self):
        """
        B coefficient matrix.

        Returns
        -------
        torch.Tensor
        """
        return torch.softmax(self._B, dim=1)

    @property
    def Z(self):
        """
        The archetypes matrix.

        Returns
        -------
        torch.Tensor
        """
        return self._Z
