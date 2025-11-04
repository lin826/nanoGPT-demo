"""Main module to demonstrate encoding and decoding of strings using unique characters."""

import torch

from train.transformer import Transformer

if __name__ == "__main__":
    transformer = Transformer()
    for step in range(10000):
        loss_item = transformer.train_batch()
    print(loss_item)

    # Generate new tokens
    idx = torch.zeros((1, 1), dtype=torch.long)
    predictions = transformer.model.generate(idx=idx, max_new_tokens=500)
    result_ints = predictions[0].tolist()
    print(transformer.decode(result_ints))
