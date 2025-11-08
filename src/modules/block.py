"""A Transformer block consisting of self-attention and feed-forward layers."""

import torch
from torch import nn

from src.modules.feed_forward import FeedForward
from src.modules.self_attention.multi_head import MultiHeadSelfAttention


class Block(nn.Module):
    '''A Transformer block consisting of self-attention and feed-forward layers.'''
    def __init__(
        self,
        block_size: int,
        device: str,
        number_of_embedding_dimensions: int = 32,
        self_attension_dimmensions: int = 4,
    ):
        super().__init__()
        self.self_attension_head = MultiHeadSelfAttention(
            num_heads=self_attension_dimmensions,
            block_size=block_size,
            channels=number_of_embedding_dimensions,
            device=device,
            head_size=number_of_embedding_dimensions // self_attension_dimmensions,
        )
        self.feed_forward = FeedForward(
            input_dim=number_of_embedding_dimensions,
            hidden_dim=number_of_embedding_dimensions * 4,
            device=device,
        )
        self.layered_norm_1 = nn.LayerNorm(number_of_embedding_dimensions).to(device)
        self.layered_norm_2 = nn.LayerNorm(number_of_embedding_dimensions).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Performs a forward pass of the block.'''
        # Residual connections by the skip connection
        # x = x + self.self_attension_head.forward(self.layered_norm_1(x))
        # x = x + self.feed_forward.forward(self.layered_norm_2(x))
        x = self.self_attension_head.forward(x)
        x = self.feed_forward.forward(x)
        return x
