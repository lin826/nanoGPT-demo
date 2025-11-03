"""Main module to demonstrate encoding and decoding of strings using unique characters."""

from model.blm import BigramLanguageModel
from train.transformer import Transformer


if __name__ == "__main__":
    transformer = Transformer()
    x_batch, y_batch = transformer.train_batch()
    vocab_size = transformer.get_vocab_size()

    model = BigramLanguageModel(vocab_size)
    logits, loss = model(x_batch, y_batch)
    print(logits.shape)
    print(loss)
