"""A data parser for preparing training and validation datasets."""
import torch

MANUAL_SEED = 1337

ContextTargetPair = tuple[torch.Tensor, torch.dtype]
BatchBlocks = list[list[ContextTargetPair]]


class DataParser:
    '''A class to parse tensor data into training and validation datasets.'''
    def __init__(
        self,
        tensor_data: torch.Tensor,
        train_val_ratio = 0.9,
        block_size = 8,
        batch_size = 4,
    ):
        # Why do we need the batch_size and block_size the same from train to validation?
        self._block_size = block_size
        self._batch_size = batch_size

        torch.manual_seed(MANUAL_SEED)
        self._update_train_val(tensor_data, train_val_ratio)

    def sample_training_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        '''Returns a batch of training data as context-target pairs.'''
        return self._process_batch(self.train_data)

    def _process_batch(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_indices = self._get_batch(data)
        x_batch = torch.stack([data[i:i+ self._block_size] for i in batch_indices])
        y_batch = torch.stack([data[i+1:i+ self._block_size + 1] for i in batch_indices])
        # return self._get_context_target(x_batch, y_batch)
        return x_batch, y_batch

    def _get_batch(self, data) -> torch.Tensor:
        return torch.randint(len(data) - self._block_size, (self._batch_size,))

    def _get_context_target(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> BatchBlocks:
        return list(map(
            lambda i: list(map(
                lambda j: (x_batch[i, :j+1], y_batch[i, j]),
                range(self._block_size))),
            range(self._batch_size)
        ))

    def _update_train_val(self, data: torch.Tensor, train_val_ratio: float) -> None:
        n = int(train_val_ratio * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
