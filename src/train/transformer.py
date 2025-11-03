"""A simple Transformer model implementation."""
from utils.data_parser import DataParser
from utils.input_converter import InputConverter
from utils.input_loader import InputLoader


class Transformer:
    '''A simple Transformer model placeholder.'''
    def __init__(self):
        input_string: str = InputLoader().parse()
        self._converter = InputConverter(input_string)

        tnesor = self._converter.get_tensor()
        self._data_parser = DataParser(tnesor)

    def get_vocab_size(self) -> int:
        '''Returns the size of the vocabulary used by the model.'''
        return self._converter.get_vocab_size()

    def train_batch(self):
        '''Processes a training batch.'''
        return self._data_parser.process_train()
