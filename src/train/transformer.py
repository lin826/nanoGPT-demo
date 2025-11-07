"""A simple Transformer model implementation."""

import torch
from torch import nn

from src.modules.bigram import BigramLanguageModel
from src.utils.data_parser import DataParser
from src.utils.input_converter import InputConverter
from src.utils.input_loader import InputLoader


class Transformer:
    '''A simple Transformer model implementation.'''
    def __init__(
        self,
        block_size: int = 8,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
        train_val_ratio: float = 0.9,
        device = 'cpu',
        model_type: nn.Module = BigramLanguageModel,
        optimizer_type: torch.optim.Optimizer = torch.optim.AdamW,
    ):
        input_string: str = InputLoader().parse()
        self._converter = InputConverter(input_string)

        tnesor = self._converter.get_tensor()
        self._data_parser = DataParser(
            tnesor, train_val_ratio, block_size, batch_size, device)

        self.model = self._get_model(model_type, block_size, device)
        self._optimizer = optimizer_type(self.model.parameters(), lr=learning_rate)

    def train_batch(self):
        '''Processes a training batch.'''
        loss = self._calculate_loss(self._data_parser.sample_training_data)
        self._optimizer.zero_grad(set_to_none=True)

        loss.backward()
        self._optimizer.step()

        return loss.item()

    def decode(self, int_list: list[int]) -> str:
        '''Decodes a list of integers back into a string.'''
        return self._converter.decode(int_list)

    @torch.no_grad()
    def estimate_losses(self) -> tuple[float, float]:
        '''Estimates training and validation loss.'''
        self.model.eval()
        training_loss = self._get_loss_mean(self._data_parser.sample_training_data)
        validate_loss = self._get_loss_mean(self._data_parser.sample_validation_data)
        self.model.train()
        return training_loss, validate_loss

    def _get_model(self, model_type: nn.Module, block_size: int, device: str) -> None:
        vocab_size = self._converter.get_vocab_size()
        return model_type(vocab_size, block_size, device).to(device)

    def _calculate_loss(self, data_sampler: callable) -> torch.Tensor:
        x_batch, y_batch = data_sampler()
        _, loss = self.model(x_batch, y_batch)
        return loss

    def _get_loss_mean(self, data_sampler: callable, iteration_amount = 1000) -> float:
        losses = torch.zeros(iteration_amount)
        losses.apply_(lambda _:  self._calculate_loss(data_sampler).item())
        return losses.mean()
