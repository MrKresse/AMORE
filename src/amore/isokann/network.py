"""
Neural network architectures for multi-dimensional ISOKANN.

ChiNetMulti learns k membership functions chi_1,...,chi_k mapping the state
space to the probability simplex Delta^{k-1}:
    chi_i >= 0,   sum_i chi_i = 1   for all x.

The softmax output enforces the simplex constraint, making the chi functions
directly interpretable as fuzzy state assignments (PCCA+ memberships).
"""

import torch
import torch.nn as nn


class ChiNetMulti(nn.Module):
    """
    MLP mapping state-space coordinates to k simplex-valued chi functions.

    Parameters
    ----------
    in_dim : int
        Input dimensionality (e.g. 2 for 2D toy systems, 40 for PCA space).
    k : int
        Number of metastable states / Koopman eigenfunctions.
    hidden : list[int]
        Hidden layer widths.
    dropout : float
        Dropout probability (0 = disabled). Useful for high-dim inputs.
    """

    def __init__(self, in_dim: int, k: int, hidden: list[int] = (128, 64, 32),
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.k = k
        dims = [in_dim] + list(hidden) + [k]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.trunk   = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, in_dim)

        Returns
        -------
        chi : (batch, k)  — rows sum to 1, all entries >= 0.
        """
        return self.softmax(self.trunk(x))


class ChiNetMultiRaw(nn.Module):
    """
    Unconstrained variant: k independent sigmoid outputs in (0,1).

    Useful when you do NOT want to impose the simplex constraint
    (e.g. for the first few eigenfunctions of a non-reversible system
    where PCCA+ is not applicable).  The outputs are orthogonalised
    during training, not by architecture.
    """

    def __init__(self, in_dim: int, k: int,
                 hidden: list[int] = (128, 64, 32)) -> None:
        super().__init__()
        dims = [in_dim] + list(hidden) + [k]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
