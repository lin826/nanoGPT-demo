"""A simple Transformer model implementation."""
import torch
import torch.nn as nn

from model.blm import BigramLanguageModel
from utils.data_parser import DataParser
from utils.input_converter import InputConverter
from utils.input_loader import InputLoader


class Transformer:
    '''A simple Transformer model placeholder.'''
    def __init__(
        self,
        model_type: nn.Module = BigramLanguageModel,
        optimizer_type: torch.optim.Optimizer = torch.optim.AdamW,
        learning_rate: float = 1e-3,
    ):
        input_string: str = InputLoader().parse()
        self._converter = InputConverter(input_string)

        tnesor = self._converter.get_tensor()
        self._data_parser = DataParser(tnesor)

        self.model = self._get_model(model_type)
        self._optimizer = optimizer_type(self.model.parameters(), lr=learning_rate)

    def train_batch(self):
        '''Processes a training batch.'''
        x_batch, y_batch = self._data_parser.sample_training_data()

        _, loss = self.model(x_batch, y_batch)
        self._optimizer.zero_grad(set_to_none=True)

        loss.backward()
        self._optimizer.step()

        return loss.item()

    def decode(self, int_list: list[int]) -> str:
        '''Decodes a list of integers back into a string.'''
        return self._converter.decode(int_list)

    def _get_model(self, model_type: nn.Module) -> None:
        vocab_size = self._converter.get_vocab_size()
        return model_type(vocab_size)
