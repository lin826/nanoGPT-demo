"""A simple Bigram Language Model implementation."""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional

MANUAL_SEED = 1337

Logits = torch.Tensor
Loss = torch.Tensor


class BigramLanguageModel(nn.Module):
    '''A simple Bigram Language Model placeholder.'''
    def __init__(self, vocab_size: int):
        torch.manual_seed(MANUAL_SEED)
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> tuple[Logits, Loss]:
        '''Performs a forward pass of the model.'''
        logits = self.token_embedding_table(idx)

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
            # prediction trick of nn.Module
            logits, _ = self(idx)

            # the last time step of the block
            logits = logits[:, -1, :]  # (batch_size, channels)
            probs = functional.softmax(logits, dim=-1)  # (batch_size, channels)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, block_index_t+1)
        return idx
