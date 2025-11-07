"""Implements single-head self-attention mechanism."""

import torch
from torch import nn

from src.modules.self_attention.attention_base import SelfAttentionBase

class SingleHeadSelfAttention(SelfAttentionBase):
    '''Implements single-head self-attention mechanism.'''
    def __init__(
        self,  block_size: int, channels: int, device: str, head_size: int, dropout: float
    ):
        super().__init__(block_size, channels, device, head_size)
        self.block_square_shape = (block_size, block_size)
        self.key = nn.Linear(channels, head_size, bias=False)
        self.query = nn.Linear(channels, head_size, bias=False)
        self.value = nn.Linear(channels, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        '''Computes the self-attention weighted aggregation.'''
        keys = self.key(x_batch)   # (batch_size, block_size, head_size)
        queries = self.query(x_batch)  # (batch_size, block_size, head_size)

        # compute attention scores
        weight = queries @ keys.transpose(-2, -1) * self.channels ** -0.5
        scores = weight.masked_fill(
            self.tril[:self.block_size, :self.block_size] == 0, float('-inf'))
        # scores = torch.matmul(queries, keys.transpose(-2, -1)) / (keys.shape[-1] ** 0.5)

        # apply softmax to get attention weights
        weights = torch.nn.functional.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # weighted sum of values (here values are the same as x_batch for simplicity)
        return torch.matmul(weights, self.value(x_batch))
