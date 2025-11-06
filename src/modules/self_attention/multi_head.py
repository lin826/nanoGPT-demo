"""Implements multi-head self-attention mechanism."""
import torch
import torch.nn as nn

from modules.self_attention.attention_base import SelfAttentionBase
from modules.self_attention.single_head import SingleHeadSelfAttention

class MultiHeadSelfAttention(SelfAttentionBase):
    '''Implements multi-head self-attention mechanism.'''
    def __init__(self, num_heads: int, block_size: int, channels: int, device: str, head_size: int):
        super().__init__(block_size, channels, device, head_size)
        self.heads = nn.ModuleList(
            [SingleHeadSelfAttention(block_size, channels, device, head_size)] * num_heads
        )

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        '''Computes the multi-head self-attention weighted aggregation.'''
        head_outputs = list(map(lambda head: head(x_batch), self.heads))
        return torch.cat(head_outputs, dim=-1)
