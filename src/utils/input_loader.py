"""Utility module for parsing input data from text files."""
import logging
from pathlib import Path


DEFAULT_INPUT_PATH = './data/inputs/'

class InputLoader:
    '''A class to parse input data from text files in a specified directory.'''
    def __init__(
        self,
        logger=logging.getLogger(__name__),
    ):
        self.logger = logger  # TODO: Implement leveled logging functionality

    def parse(self, directory_path: Path=Path(DEFAULT_INPUT_PATH)) -> str:
        '''Parses input data from text files in the specified directory.'''
        if not directory_path.exists():
            self.logger.error("Directory %s does not exist.", directory_path)
            return ''

        input_data = []
        for item in directory_path.iterdir():
            if not item.is_file():
                self.logger.debug("Skipping non-file item: %s", item)
                continue
            if item.name.startswith('_'):
                self.logger.debug("Skipping hidden/system file: %s", item)
                continue
            if item.suffix != '.txt':
                self.logger.debug("Skipping non-text file: %s", item)
                continue
            with item.open('r') as file:
                input_data += file.read()
        return input_data
