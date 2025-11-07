"""Base class for self-attention mechanisms."""

import torch
from torch import nn

class SelfAttentionBase(nn.Module):
    '''Base class for self-attention mechanisms.'''
    def __init__(self, block_size: int, channels: int, device: str, head_size: int):
        super().__init__()
        self.channels = channels
        self.device = device
        self.head_size = head_size
        self.block_size = block_size

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        '''Placeholder forward method to be overridden by subclasses.'''
        raise NotImplementedError("Subclasses should implement this method.")
