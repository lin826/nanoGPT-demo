"""Implements multi-head self-attention mechanism."""
import torch
import torch.nn as nn

from src.modules.self_attention.attention_base import SelfAttentionBase
from src.modules.self_attention.single_head import SingleHeadSelfAttention

class MultiHeadSelfAttention(SelfAttentionBase):
    '''Implements multi-head self-attention mechanism.'''
    def __init__(
        self,
        num_heads: int,
        block_size: int,
        channels: int,
        device: str,
        head_size: int,
        dropout: float = 0.1,
    ):
        super().__init__(block_size, channels, device, head_size)
        self.heads = nn.ModuleList(
            [SingleHeadSelfAttention(block_size, channels, device, head_size, dropout)] * num_heads
        )
        self.projection = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        '''Computes the multi-head self-attention weighted aggregation.'''
        head_outputs = list(map(lambda head: head(x_batch), self.heads))
        formatted_output = torch.cat(head_outputs, dim=-1)
        dropped_output = self.dropout(formatted_output)
        return self.projection(dropped_output)
