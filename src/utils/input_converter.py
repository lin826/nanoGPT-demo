"""A module for data handling and conversion between string and integer representations."""

import torch


class InputConverter:
    '''A class for handling string data and converting it to and from integer representations.'''
    def __init__(self, input_str: str):
        self._input = input_str

        self._update_mappings()
        self._update_tensor()

    def get_input(self) -> str:
        '''Returns the original input string.'''
        return self._input

    def get_tensor(self) -> torch.Tensor:
        '''Returns the tensor representation of the input string.'''
        return self._tensor

    def get_vocab_size(self) -> int:
        '''Returns the size of the vocabulary (number of unique characters).'''
        return len(self._int_to_str_mapping)

    def encode(self, s: str) -> list[int]:
        '''Encodes a string into a list of integers based on character mappings.'''
        return list(map(self._str_to_int, s))

    def decode(self, int_list: list[int]) -> str:
        '''Decodes a list of integers back into a string based on character mappings.'''
        return ''.join(map(self._int_to_str, int_list))

    def _get_unique_chars(self) -> str:
        '''Returns a string of unique characters from the input text, sorted in order.'''
        return ''.join(sorted(list(set(self._input))))

    def _str_to_int(self, ch: str) -> int:
        return self._str_to_int_mapping[ch]

    def _int_to_str(self, i: int) -> str:
        return self._int_to_str_mapping[i]

    def _update_mappings(self) -> None:
        unique_chars = self._get_unique_chars()
        self._str_to_int_mapping = { ch:i for i,ch in enumerate(unique_chars) }
        self._int_to_str_mapping = { i:ch for i,ch in enumerate(unique_chars) }

    def _update_tensor(self) -> None:
        self._tensor = torch.tensor(self.encode(self._input), dtype=torch.long)
