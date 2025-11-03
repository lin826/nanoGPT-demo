"""Utility module for parsing input data from text files."""
import logging
from pathlib import Path

SEPERATOR = '\n\n\n'
DEFAULT_INPUT_PATH = Path('../data/inputs')

class InputParser:
    '''A class to parse input data from text files in a specified directory.'''
    def __init__(self, logger=logging.getLogger(__name__)):
        self.input_data = []
        self.logger = logger # TODO: Implement leveled logging functionality

    def parse(self, directory_path: Path = DEFAULT_INPUT_PATH) -> str:
        '''Parses input data from text files in the specified directory.'''
        if not directory_path.exists():
            self.logger.error("Directory %s does not exist.", directory_path)
            return ''
        for item in directory_path.iterdir():
            if not item.is_file():
                continue
            if item.name.startswith('_'):
                continue
            if item.suffix != '.txt':
                continue
            with item.open('r') as file:
                input_data = file.read()
            if isinstance(input_data, str):
                self.input_data += self._parse_string(input_data)
        return SEPERATOR.join(self.input_data)

    def get_unique_chars(self, text: str) -> str:
        '''Returns a string of unique characters from the input text, sorted in order.'''
        return ''.join(sorted(list(set(text))))

    def _parse_string(self, data):
        # Example parsing logic for string input
        return data.strip().split(',')

    # def _parse_list(self, data):
    #     # Example parsing logic for list input
    #     return [str(item).strip() for item in data]


if __name__ == "__main__":
    print(f"Current folder: {Path.cwd()}\n============================")

    DATA = InputParser().parse()
    unique_chars = InputParser().get_unique_chars(DATA)
    print(f"Unique characters ({len(unique_chars)}): {unique_chars}")
