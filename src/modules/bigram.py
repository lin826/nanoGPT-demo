"""A simple Bigram Language Model implementation."""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional

from src.modules.feed_forward import FeedForward
from src.modules.self_attention.multi_head import MultiHeadSelfAttention

MANUAL_SEED = 1337

Logits = torch.Tensor
Loss = torch.Tensor

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
        x = x + self.self_attension_head.forward(self.layered_norm_1(x))
        x = x + self.feed_forward.forward(self.layered_norm_2(x))
        return x


class BigramLanguageModel(nn.Module):
    '''A simple Bigram Language Model placeholder.'''
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        device: str,
        number_of_embedding_dimensions: int = 32,
        self_attension_dimmensions: int = 4,
    ):
        torch.manual_seed(MANUAL_SEED)
        super().__init__()

        self._block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, number_of_embedding_dimensions)
        self.position_embedding_table = nn.Embedding(block_size, number_of_embedding_dimensions)
        self.language_modeling_head = nn.Linear(number_of_embedding_dimensions, vocab_size)

        self.blocks = nn.Sequential(
            Block(block_size, device, number_of_embedding_dimensions, self_attension_dimmensions),
            Block(block_size, device, number_of_embedding_dimensions, self_attension_dimmensions),
            Block(block_size, device, number_of_embedding_dimensions, self_attension_dimmensions),
            nn.LayerNorm(number_of_embedding_dimensions).to(device),
        )
        self.feed_forward = FeedForward(
            input_dim=number_of_embedding_dimensions,
            hidden_dim=number_of_embedding_dimensions * 4,
            device=device,
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> tuple[Logits, Loss]:
        '''Performs a forward pass of the model.'''
        idx_position = torch.arange(idx.shape[1], device=idx.device)
        position_embedding = self.position_embedding_table(idx_position)
        token_embeddings = self.token_embedding_table(idx)
        x = token_embeddings + position_embedding

        # Communication
        x = self.blocks(x)

        # Computation
        logits = self.language_modeling_head(x)

        if targets is None:
            return logits, None

        # reshape for loss computation
        batch_size, block_size, channels = logits.shape
        minibatch = batch_size*block_size
        logits = logits.view(minibatch, channels)
        targets = targets.view(minibatch)
        # expecting the loss to be -ln(1/minibatch), why?
        loss = functional.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        '''Generates new tokens given a starting context.'''
        for _ in range(max_new_tokens):
            # crop context to the last block_size tokens
            idx_cond = idx[:, -self._block_size:]

            # prediction trick of nn.Module
            logits, _ = self(idx_cond)

            # the last time step of the block
            logits = logits[:, -1, :]  # (batch_size, channels)
            probs = functional.softmax(logits, dim=-1)  # (batch_size, channels)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, block_index_t+1)
        return idx
