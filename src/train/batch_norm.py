"""Implements 1D Batch Normalization layer."""

import torch
from torch import nn


class BatchNorm1d:
    '''Implements 1D Batch Normalization.'''

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        device: torch.device = torch.device('cpu'),
    ):
        self.num_features = num_features
        self.eps = eps
        self.device = device

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features, device=device))
        self.beta = nn.Parameter(torch.zeros(num_features, device=device))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        '''Performs a forward pass of batch normalization.'''
        mean = x.mean(1, keepdim=True)  # Mean over batch dimension
        var = x.var(1, keepdim=True)  # Variance over batch dimension

        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta
