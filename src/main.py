"""Main module to demonstrate encoding and decoding of strings using unique characters."""

import torch

from train.transformer import Transformer

BLOCK_SIZE = 8

if __name__ == "__main__":
    transformer = Transformer(block_size=BLOCK_SIZE)
    for step in range(10000):
        if step % 1000 == 0:
            training_loss, validate_loss = transformer.estimate_losses()
            print(f"Step {step:04d}: {training_loss:.4f}, {validate_loss:.4f}")
        loss_item = transformer.train_batch()
    print(f"Final loss: {loss_item:.4f}")

    # Generate new tokens
    idx = torch.zeros((1, BLOCK_SIZE), dtype=torch.long)
    predictions = transformer.model.generate(idx=idx, max_new_tokens=500)
    result_ints = predictions[0].tolist()
    print(transformer.decode(result_ints))
