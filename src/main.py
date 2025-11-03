"""Main module to demonstrate encoding and decoding of strings using unique characters."""

from pathlib import Path

from utils.data_parser import DataParser
from utils.input_converter import InputConverter
from utils.input_loader import InputLoader


if __name__ == "__main__":
    print(f"Current folder: {Path.cwd()}\n============================")

    # Parse the first text files in the default input directory
    input_string: str = InputLoader().parse()
    sample_data = InputConverter(input_string)

    # # Test encoding and decoding
    # ints = sample_data.encode("hii there")
    # print(f"Encoded integer list: {ints}")
    # decoded_result = sample_data.decode(ints)
    # print(f"Decoded string: {decoded_result}\n")

    # Display tensor representation
    tnesor = sample_data.get_tensor()
    print(f"Tensor representation: {tnesor.shape} {tnesor.dtype}\n{tnesor[:100]}\n")

    transformer_model = DataParser(tnesor)
    batch = transformer_model.process_train()
