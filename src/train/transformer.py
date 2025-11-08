"""A simple Transformer model implementation."""

import torch
from torch import nn

from src.utils.data_parser import DataParser


class Transformer:
    '''A simple Transformer model implementation.'''
    def __init__(
        self,
        data_parser: DataParser,
        model: nn.Module,
        learning_rate: float = 1e-3,
        eval_iters: int = 200,
        device = 'cpu',
        optimizer_type: torch.optim.Optimizer = torch.optim.AdamW,
    ):
        self._data_parser = data_parser
        self._model = model.to(device)
        self._eval_iters = eval_iters

        self._optimizer = optimizer_type(self._model.parameters(), lr=learning_rate)

    def train_batch(self):
        '''Processes a training batch.'''
        loss = self._calculate_loss(self._data_parser.sample_training_data)
        self._optimizer.zero_grad(set_to_none=True)

        loss.backward()
        self._optimizer.step()

        return loss.item()

    @torch.no_grad()
    def estimate_losses(self) -> tuple[float, float]:
        '''Estimates training and validation loss.'''
        self._model.eval()
        training_loss = self._get_loss_mean(self._data_parser.sample_training_data)
        validate_loss = self._get_loss_mean(self._data_parser.sample_validation_data)
        self._model.train()
        return training_loss, validate_loss

    def _calculate_loss(self, data_sampler: callable) -> torch.Tensor:
        x_batch, y_batch = data_sampler()
        _, loss = self._model(x_batch, y_batch)
        return loss

    def _get_loss_mean(self, data_sampler: callable) -> float:
        losses = torch.zeros(self._eval_iters)
        for k in range(self._eval_iters):
            losses[k] = self._calculate_loss(data_sampler).item()
        return losses.mean()
