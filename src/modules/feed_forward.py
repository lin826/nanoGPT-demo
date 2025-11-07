"""A simple feed-forward neural network layer."""
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    '''A simple feed-forward neural network layer.'''
    def __init__(self, input_dim: int, hidden_dim: int, device: str, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),  # project back to residual pathway
            nn.Dropout(dropout),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Performs a forward pass of the feed-forward network.'''
        return self.net(x)
