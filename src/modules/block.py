"""A Transformer block consisting of self-attention and feed-forward layers."""

from typing import Literal
import torch
from torch import nn

from src.modules.feed_forward import FeedForward
from src.modules.self_attention.multi_head import MultiHeadSelfAttention


class Block(nn.Module):
    '''A Transformer block: communication followed by computation.'''
    def __init__(
        self,
        block_size: int,
        device: Literal["cpu", "cuda", "mps"],
        number_of_embedding_dimensions: int,
        self_attension_dimmensions: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attension_head = MultiHeadSelfAttention(
            block_size=block_size,
            number_of_embedding_dimensions=number_of_embedding_dimensions,
            head_size=self_attension_dimmensions,
            dropout=dropout,  # no dropout
        )
        self.feed_forward = FeedForward(
            input_dim=number_of_embedding_dimensions,
            hidden_dim=number_of_embedding_dimensions * 4,  # as suggested in the paper
            device=device,
            dropout=dropout,  # no dropout
        )
        self.layered_norm_1 = nn.LayerNorm(number_of_embedding_dimensions).to(device)
        self.layered_norm_2 = nn.LayerNorm(number_of_embedding_dimensions).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Performs a forward pass of the block.'''
        # Residual connections by the skip connection
        x = x + self.self_attension_head.forward(self.layered_norm_1(x))
        x = x + self.feed_forward.forward(self.layered_norm_2(x))
        # x = x + self.self_attension_head.forward(x)
        # x = x + self.feed_forward.forward(x)
        return x
