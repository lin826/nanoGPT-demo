"""Utility module for preprocessing string data into the tensor representation."""

import logging
from pathlib import Path
import torch

DEFAULT_INPUT_PATH = './data/inputs/'


class InputConverter:
    '''A class for handling string data and converting it to and from integer representations.'''

    _str_to_int_mapping: dict[str, int]
    _int_to_str_mapping: dict[int, str]
    _input_tensor: torch.Tensor

    def __init__(
        self,
        directory_path: Path=Path(DEFAULT_INPUT_PATH),
        logger = logging.getLogger(__name__),
    ):
        self._logger = logger
        self._input = self._load(directory_path)

        self._update_mappings()
        self._update_tensor()

    def get_input(self) -> str:
        '''Returns the original input string.'''
        return self._input

    def get_input_tensor(self) -> torch.Tensor:
        '''Returns the tensor representation of the input string.'''
        return self._input_tensor

    def get_vocab_size(self) -> int:
        '''Returns the size of the vocabulary (number of unique characters).'''
        return len(self._int_to_str_mapping)

    def encode(self, s: str) -> list[int]:
        '''Encodes a string into a list of integers based on character mappings.'''
        return list(map(self._str_to_int, s))

    def decode(self, int_list: list[int]) -> str:
        '''Decodes a list of integers back into a string based on character mappings.'''
        return ''.join(map(self._int_to_str, int_list))

    def _get_unique_chars(self) -> list[str]:
        '''Returns a string of unique characters from the input text, sorted in order.'''
        return sorted(list(set(self._input)))

    def _str_to_int(self, ch: str) -> int:
        return self._str_to_int_mapping[ch]

    def _int_to_str(self, i: int) -> str:
        return self._int_to_str_mapping[i]

    def _update_mappings(self) -> None:
        unique_chars = self._get_unique_chars()
        self._str_to_int_mapping = { ch:i for i,ch in enumerate(unique_chars) }
        self._int_to_str_mapping = { i:ch for i,ch in enumerate(unique_chars) }

    def _update_tensor(self) -> None:
        self._input_tensor = torch.tensor(self.encode(self._input), dtype=torch.long)

    def _load(self, directory_path: Path=Path(DEFAULT_INPUT_PATH)) -> str:
        '''Load input data from text files in the specified directory.'''
        if not directory_path.exists():
            self._logger.error("Directory %s does not exist.", directory_path)
            return ''

        input_data = ''
        for item in directory_path.iterdir():
            if not item.is_file():
                self._logger.debug("Skipping non-file item: %s", item)
                continue
            if item.name.startswith('_'):
                self._logger.debug("Skipping hidden/system file: %s", item)
                continue
            if item.suffix != '.txt':
                self._logger.debug("Skipping non-text file: %s", item)
                continue
            with item.open('r') as file:
                input_data += file.read()
        return input_data
