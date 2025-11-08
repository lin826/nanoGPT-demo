"""Main module to demonstrate encoding and decoding of strings using unique characters."""

import torch

from src.modules.bigram import BigramLanguageModel
from src.train.transformer import Transformer
from src.utils.data_parser import DataParser
from src.utils.input_converter import InputConverter
from src.utils.input_loader import InputLoader

BATCH_SIZE = 32
BLOCK_SIZE = 8
LEARNING_RATE = 1e-2
TRAIN_VAL_RATIO = 0.9
DEVICE = 'cpu'

NUMBER_OF_EMBEDDING_DIMENSIONS = 384
SELF_ATTENTION_DIMENSIONS = 6
BLOCK_LAYERS = 6
DROPOUT = 0.0

MAX_ITERS = 3000
EVAL_INTERVAL = 300
EVAL_ITERS = 200

TORCH_SEED = 1337

# TODO: Argparse for command line arguments

def main():
    """Main function to demonstrate encoding and decoding of strings."""
    input_string: str = InputLoader().parse()
    converter = InputConverter(input_string)

    data_parser = DataParser(
        converter.get_tensor(),
        TRAIN_VAL_RATIO,
        BLOCK_SIZE,
        BATCH_SIZE,
        DEVICE,
        TORCH_SEED
    )
    model = BigramLanguageModel(
        converter.get_vocab_size(),
        BLOCK_SIZE,
        DEVICE,
        NUMBER_OF_EMBEDDING_DIMENSIONS,
        SELF_ATTENTION_DIMENSIONS,
    )

    transformer = Transformer(
        data_parser=data_parser,
        learning_rate=LEARNING_RATE,
        eval_iters=EVAL_ITERS,
        device=DEVICE,
        model=model,
    )
    for step in range(MAX_ITERS):
        if step % EVAL_INTERVAL == 0:
            training_loss, validate_loss = transformer.estimate_losses()
            print(f"Step {step:04d}: {training_loss:.4f}, {validate_loss:.4f}")
        loss_item = transformer.train_batch()
    print(f"Final loss: {loss_item:.4f}")

    # Generate new tokens
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    predictions = model.generate(idx=context, max_new_tokens=500)
    result_ints = predictions[0].tolist()
    print(converter.decode(result_ints))

if __name__ == "__main__":
    main()
