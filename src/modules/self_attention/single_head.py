"""Implements single-head self-attention mechanism."""

import torch
from torch import nn

from src.modules.self_attention.attention_base import SelfAttentionBase

class SingleHeadSelfAttention(SelfAttentionBase):
    '''Implements single-head self-attention mechanism.'''
    def __init__(
        self,  block_size: int, channels: int, head_size: int, dropout: float
    ):
        super().__init__(block_size, channels, head_size)
        self.block_square_shape = (block_size, block_size)
        self.key = nn.Linear(channels, head_size, bias=False)
        self.query = nn.Linear(channels, head_size, bias=False)

        # Corrected from head_size to channels as the output dimension
        self.value = nn.Linear(channels, head_size, bias=False)

        self.tril: torch.Tensor  # resolve Pylance warning
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        '''Computes the self-attention weighted aggregation.'''
        keys = self.key(x_batch)   # (batch_size, block_size, head_size)
        queries = self.query(x_batch)  # (batch_size, block_size, head_size)

        # Using einsum to speed up: (B,T,C) x (B,C,T) -> (B,T,T)
        weight = torch.einsum('bth,bsh->bts', queries, keys)
        weight = weight * self.channels ** -0.5  # scale with normalization
        # use registered lower-triangular mask tensor directly
        scores = weight.masked_fill(self.tril == 0, float('-inf'))
        # scores = torch.einsum('bts,bsh->bth', queries, keys.transpose(-2, -1))
        # scores = scores / (keys.shape[-1] ** 0.5)

        # apply softmax to get attention weights
        weights = torch.nn.functional.softmax(scores, dim=-1)
        # weights = self.dropout(weights)

        # weighted sum of values: (B,T,T) x (B,T,C) -> (B,T,C)
        # Here values are the same as x_batch for simplicity
        v = torch.einsum('bts,bsh->bth', weights, self.value(x_batch))
        return v
