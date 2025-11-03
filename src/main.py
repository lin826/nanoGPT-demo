"""Main module to demonstrate encoding and decoding of strings using unique characters."""

from pathlib import Path

from utils.input_data import InputData
from utils.input_parser import InputParser


if __name__ == "__main__":
    print(f"Current folder: {Path.cwd()}\n============================")

    # Parse the first text files in the default input directory
    input_strings: list[str] = InputParser().parse()
    sample_data = InputData(input_strings[0])

    # Test encoding and decoding
    ints = sample_data.encode("hii there")
    print(f"Encoded integer list: {ints}")
    decoded_result = sample_data.decode(ints)
    print(f"Decoded string: {decoded_result}\n")

    # Display tensor representation
    tnesor = sample_data.get_tensor()
    print(f"Tensor representation: {tnesor.shape} {tnesor.dtype}\n{tnesor[:100]}\n")
