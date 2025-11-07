"""Tests for the BigramLanguageModel module."""

from src.modules.bigram import BigramLanguageModel
from src.utils.data_parser import DataParser
from src.utils.input_converter import InputConverter
from src.utils.input_loader import InputLoader

def test_bigram_initialization():
    '''Tests the initialization of the BigramLanguageModel.'''
    # Arrange
    input_string: str = InputLoader().parse()
    converter = InputConverter(input_string)

    tnesor = converter.get_tensor()
    data_parser = DataParser(tnesor)
    vocab_size = converter.get_vocab_size()
    x_batch, y_batch = data_parser.sample_training_data()

    # Act
    model = BigramLanguageModel(vocab_size, block_size=8, device='cpu')
    logits, loss = model(x_batch, y_batch)

    # Assert
    assert logits.shape == (x_batch.size(0)*x_batch.size(1), vocab_size)
    assert loss > 0
