"""Implements 1D Batch Normalization layer."""
import torch
import torch.nn as nn


class BatchNorm1d:
    '''Implements 1D Batch Normalization.'''

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device: torch.device = torch.device('cpu'),
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.device = device

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features, device=device))
        self.beta = nn.Parameter(torch.zeros(num_features, device=device))

        # Running estimates
        self.running_mean = torch.zeros(num_features, device=device)
        self.running_var = torch.ones(num_features, device=device)

    def __call__(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        '''Performs a forward pass of batch normalization.'''
        if training:
            mean = x.mean(0, keepdim=True)  # Mean over batch dimension
            var = x.var(0, keepdim=True)  # Variance over batch dimension
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * x_normalized + self.beta

        if training:
            with torch.no_grad():
                # Update running estimates
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        return out
