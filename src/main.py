"""Main module to demonstrate encoding and decoding of strings using unique characters."""

from typing import Literal
import torch

from src.modules.bigram import BigramLanguageModel
from src.train.transformer import Transformer
from src.utils.data_parser import DataParser
from src.utils.input_loader import InputLoader

LEARNING_RATE = 3e-4
TRAIN_VAL_RATIO = 0.9
DEVICE: Literal["cpu", "cuda", "mps"] = 'mps'

BATCH_SIZE = 64
BLOCK_SIZE = 256
NUMBER_OF_EMBEDDING_DIMENSIONS: int = 384
SELF_ATTENTION_DIMENSIONS: int = 64
NUM_HEADS: int = 6  # 6 heads of 64-dimensional self-attension
BLOCK_LAYERS = 6  # 6 layers of transformer blocks
DROPOUT = 0.2

MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200

TORCH_SEED = 1337

# TODO: Argparse for command line arguments

def main():
    """Main function to demonstrate encoding and decoding of strings."""
    assert NUMBER_OF_EMBEDDING_DIMENSIONS == NUM_HEADS * SELF_ATTENTION_DIMENSIONS

    torch.manual_seed(TORCH_SEED)
    converter = InputLoader()

    data_parser = DataParser(
        converter.get_input_tensor(),
        TRAIN_VAL_RATIO,
        BLOCK_SIZE,
        BATCH_SIZE,
        DEVICE
    )

    model = BigramLanguageModel(
        converter.get_vocab_size(),
        BLOCK_SIZE,
        DEVICE,
        NUMBER_OF_EMBEDDING_DIMENSIONS,
        SELF_ATTENTION_DIMENSIONS,
        BLOCK_LAYERS,
        DROPOUT,
    )

    transformer = Transformer(
        data_parser=data_parser,
        learning_rate=LEARNING_RATE,
        eval_iters=EVAL_ITERS,
        device=DEVICE,
        model=model,
    )

    best_model_state = model.state_dict()
    min_validate_loss = float('inf')
    for step in range(MAX_ITERS):
        if step % EVAL_INTERVAL == 0:
            training_loss, validate_loss = transformer.estimate_losses()
            print(f"Step {step:04d}: {training_loss:.4f}, {validate_loss:.4f}")
            if validate_loss < min_validate_loss:
                min_validate_loss = validate_loss
                best_model_state = model.state_dict()
        training_loss = transformer.train_batch()

    print("\n---END OF TRAINING---")
    print(f"Last training loss: {training_loss:.4f}")
    print(f"Min validation loss: {min_validate_loss:.4f}\n")

    model.load_state_dict(best_model_state)

    # Generate new tokens
    context = torch.zeros((1, BLOCK_SIZE), dtype=torch.long, device=DEVICE)
    predictions = model.generate(idx=context, max_new_tokens=500)
    result_ints = predictions[0].tolist()
    print(converter.decode(result_ints).strip())

if __name__ == "__main__":
    main()
