"""Utility module for parsing input data from text files."""
import logging
from pathlib import Path

DEFAULT_INPUT_PATH = Path('../data/inputs')

class InputParser:
    '''A class to parse input data from text files in a specified directory.'''
    def __init__(
        self,
        logger=logging.getLogger(__name__),
    ):
        self.logger = logger # TODO: Implement leveled logging functionality

    def parse(self, directory_path: Path=DEFAULT_INPUT_PATH) -> list[str]:
        '''Parses input data from text files in the specified directory.'''
        if not directory_path.exists():
            self.logger.error("Directory %s does not exist.", directory_path)
            return ''

        input_data = []
        for item in directory_path.iterdir():
            if not item.is_file():
                continue
            if item.name.startswith('_'):
                continue
            if item.suffix != '.txt':
                continue
            with item.open('r') as file:
                input_data += file.read()
        return input_data
