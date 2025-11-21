"""Tests for the BigramLanguageModel module."""

from src.modules.bigram import BigramLanguageModel
from src.utils.data_parser import DataParser
from utils.input_loader import InputLoader

def test_bigram_initialization():
    '''Tests the initialization of the BigramLanguageModel.'''
    # Arrange
    converter = InputLoader()
    tnesor = converter.get_input_tensor()

    batch_size, block_size, device = 32, 8, 'cpu'
    train_val_ratio = 0.8
    data_parser = DataParser(tnesor, train_val_ratio, block_size, batch_size, device)
    vocab_size = converter.get_vocab_size()
    x_batch, y_batch = data_parser.sample_training_data()

    n_emb, self_attn_dim = 32, 8

    # Act
    model = BigramLanguageModel(vocab_size, block_size, device, n_emb, self_attn_dim)
    logits, loss = model(x_batch, y_batch)

    # Assert
    assert logits.shape == (x_batch.size(0)*x_batch.size(1), vocab_size)
    assert loss > 0
