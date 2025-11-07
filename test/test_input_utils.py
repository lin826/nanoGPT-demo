"""Tests for the InputConverter module."""

import torch

from src.utils.data_parser import DataParser
from src.utils.input_converter import InputConverter
from src.utils.input_loader import InputLoader


def test_encoding():
    '''Tests the encoding functionality of InputConverter.'''
    input_string: str = InputLoader().parse()
    sample_data = InputConverter(input_string)
    ints = sample_data.encode("hii there")

    assert isinstance(ints, list)
    assert all(isinstance(i, int) for i in ints)

def test_decoding():
    '''Tests the decoding functionality of InputConverter.'''
    input_string: str = InputLoader().parse()
    sample_data = InputConverter(input_string)
    ints = [1, 2, 3, 4, 5]
    decoded_result = sample_data.decode(ints)

    assert isinstance(decoded_result, str)
    assert all(char in input_string for char in decoded_result)

def test_tensor_representation():
    '''Tests the tensor representation of the input string.'''
    input_string: str = InputLoader().parse()
    sample_data = InputConverter(input_string)
    tensor = sample_data.get_tensor()

    assert tensor.dim() == 1
    assert tensor.dtype == torch.long
    assert tensor.size(0) == len(input_string)

def test_data_parser():
    '''Tests the data parser integration with InputConverter.'''
    batch_size = 4
    block_size = 8
    input_string: str = InputLoader().parse()
    sample_data = InputConverter(input_string)
    tensor = sample_data.get_tensor()
    transformer_model = DataParser(
        tensor, block_size=block_size, batch_size=batch_size)

    # Act: Sample training data
    x_batch, y_batch = transformer_model.sample_training_data()

    # Assertions: Check shapes and types
    assert x_batch.size() == (batch_size, block_size)
    assert y_batch.size() == (batch_size, block_size)

    context_target_pairs = list(map(
        lambda i: list(map(
            lambda j: (x_batch[i, :j+1], y_batch[i, j]),
            range(block_size))),
        range(batch_size)
    ))
    for batch in context_target_pairs:
        for context, target in batch:
            assert context.size(0) <= block_size
            assert target.dim() == 0
            assert isinstance(target.item(), int)
