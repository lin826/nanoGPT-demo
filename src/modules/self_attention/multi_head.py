"""Implements multi-head self-attention mechanism."""

import torch
from torch import nn

from src.modules.self_attention.attention_base import SelfAttentionBase
from src.modules.self_attention.single_head import SingleHeadSelfAttention

class MultiHeadSelfAttention(SelfAttentionBase):
    '''Implements multi-head self-attention mechanism.'''
    def __init__(
        self,
        block_size: int,
        number_of_embedding_dimensions: int,
        head_size: int,
        dropout: float,
    ):
        super().__init__(block_size, number_of_embedding_dimensions, head_size)

        num_heads = number_of_embedding_dimensions // head_size
        self.heads = nn.ModuleList([
            SingleHeadSelfAttention(block_size, number_of_embedding_dimensions, head_size, dropout)
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(number_of_embedding_dimensions, number_of_embedding_dimensions)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        '''Computes the multi-head self-attention weighted aggregation.'''
        head_outputs = list(map(lambda head: head(x_batch), self.heads))
        formatted_output = torch.cat(head_outputs, dim=-1)
        dropped_output = self.dropout(formatted_output)
        return self.projection(dropped_output)
