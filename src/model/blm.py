"""A simple Bigram Language Model implementation."""
import torch
import torch.nn as nn
from torch.nn import functional

MANUAL_SEED = 1337

Logits = torch.Tensor
Loss = torch.Tensor


class BigramLanguageModel(nn.Module):
    '''A simple Bigram Language Model placeholder.'''
    def __init__(self, vocab_size: int):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

        torch.manual_seed(MANUAL_SEED)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> tuple[Logits, Loss]:
        '''Performs a forward pass of the model.'''
        logits = self.token_embedding_table(idx)

        # reshape for loss computation
        batch_size, block_size, count = logits.shape
        minibatch = batch_size*block_size
        logits = logits.view(minibatch, count)
        targets = targets.view(minibatch)
        loss = functional.cross_entropy(logits, targets)
        return logits, loss
