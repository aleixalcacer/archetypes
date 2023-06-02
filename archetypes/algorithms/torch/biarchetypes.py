import torch
import torch.nn as nn
from tqdm import tqdm


class BiAA(nn.Module):
    """
    Biarchetype analysis implemented in PyTorch.

    Parameters
    ----------
    k: tuple
        The number of archetypes to use for each dimension.

    m: int
        The number of observations in the first dimension.

    n: int
        The number of observations in the second dimension.

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
            torch.randn(self.m, self.k[0], device=self.device), requires_grad=True
        )

        self._B = torch.nn.Parameter(
            torch.randn(self.k[0], self.m, device=self.device), requires_grad=True
        )

        self._Z = torch.zeros(self.k[0], self.k[1], device=self.device)

        self._C = torch.nn.Parameter(
            torch.randn(self.n, self.k[1], device=self.device), requires_grad=True
        )

        self._D = torch.nn.Parameter(
            torch.randn(self.k[1], self.n, device=self.device), requires_grad=True
        )

        self.losses = []

    def _loss(self):
        """
        The negative log-likelihood of a normal distribution
        """
        loss = torch.pow(self._X - self.A @ self.B @ self._X @ self.C @ self.D, 2).sum()
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

        optimizer_out = torch.optim.Adam(params=[self._A, self._D], lr=learning_rate)
        optimizer_in = torch.optim.Adam(params=[self._B, self._C], lr=learning_rate)

        pbar_epoch = tqdm(range(n_epochs), leave=True)

        for epoch in pbar_epoch:
            loss = self._loss()
            optimizer_out.zero_grad()
            loss.backward()
            optimizer_out.step()

            loss = self._loss()
            optimizer_in.zero_grad()
            loss.backward()
            optimizer_in.step()

            self.losses.append(loss.item())
            pbar_epoch.set_description(f"Epoch {epoch}/{n_epochs} | loss {loss.item():.4f}")

        self._Z = self.B @ self._X @ self.C

    @property
    def A(self):
        """
        A coefficients matrix.

        Returns
        -------
        torch.Tensor
        """
        return torch.softmax(self._A, dim=1)

    @property
    def B(self):
        """
        B coefficients matrix.

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

    @property
    def C(self):
        """
        C coefficients matrix.

        Returns
        -------
        torch.Tensor
        """
        return torch.softmax(self._C, dim=0)

    @property
    def D(self):
        """
        D coefficients matrix.

        Returns
        -------
        torch.Tensor
        """
        return torch.softmax(self._D, dim=0)
