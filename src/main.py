"""Main module to demonstrate encoding and decoding of strings using unique characters."""

import torch

from src.train.transformer import Transformer

BATCH_SIZE = 32
BLOCK_SIZE = 8
LEARNING_RATE = 3e-4
TRAIN_VAL_RATIO = 0.9
DEVICE = 'cpu'

NUMBER_OF_EMBEDDING_DIMENSIONS = 384
SELF_ATTENTION_DIMENSIONS = 6
BLOCK_LAYERS = 6
DROPOUT = 0.2

MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200

# TODO: Argparse for command line arguments

def main():
    """Main function to demonstrate encoding and decoding of strings."""
    transformer = Transformer(
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        train_val_ratio=TRAIN_VAL_RATIO,
        device=DEVICE,
    )
    for step in range(MAX_ITERS):
        if step % EVAL_ITERS == 0:
            training_loss, validate_loss = transformer.estimate_losses()
            print(f"Step {step:04d}: {training_loss:.4f}, {validate_loss:.4f}")
        loss_item = transformer.train_batch()
    print(f"Final loss: {loss_item:.4f}")

    # Generate new tokens
    idx = torch.zeros((1, BLOCK_SIZE), dtype=torch.long)
    predictions = transformer.model.generate(idx=idx, max_new_tokens=500)
    result_ints = predictions[0].tolist()
    print(transformer.decode(result_ints))

if __name__ == "__main__":
    main()
