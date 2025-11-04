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
        device = 'cpu',
    ):
        # Why do we need the batch_size and block_size the same from train to validation?
        self._block_size = block_size
        self._batch_size = batch_size
        self.device = device

        torch.manual_seed(MANUAL_SEED)
        self._update_train_val(tensor_data, train_val_ratio)

    def sample_training_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        '''Returns a batch of training data as context-target pairs.'''
        return self._process_batch(self.train_data)

    def sample_validation_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        '''Returns a batch of validation data as context-target pairs.'''
        return self._process_batch(self.val_data)

    # def get_context_target(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> BatchBlocks:
    #     '''Debugging helper to get context-target pairs from a batch.'''
    #     return list(map(
    #         lambda i: list(map(
    #             lambda j: (x_batch[i, :j+1], y_batch[i, j]),
    #             range(self._block_size))),
    #         range(self._batch_size)
    #     ))

    def single_head_self_attention(self, x_batch: torch.Tensor) -> torch.Tensor:
        '''
        Get weighted aggregation by bag-of-words for self-attention with query and key.
        The query vector is where to looking for, and the key vectors are the words contained.
        '''
        # # Version 1: Explicit loops (inefficient)
        # x_bag_of_words = torch.zeros(x_batch.shape, device=self.device)
        # for i in range(self._batch_size):
        #     for j in range(self._block_size):
        #         x_bag_of_words[i, j] = torch.mean(x_batch[i, :j+1], 0)
        # return x_bag_of_words

        # # Version 2: Matrix multiplication (efficient)
        # tril = torch.tril(torch.ones((self._block_size, self._block_size), device=self.device))
        # weight /= tril.sum(1, keepdim=True)

        # Version 3: Using softmax for weighting (efficient)
        block_square_shape = (self._block_size, self._block_size)
        tril = torch.tril(torch.ones(block_square_shape, device=self.device))
        weight = torch.zeros(block_square_shape, device=self.device)
        weight = weight.masked_fill(tril == 0, float('-inf'))
        weight = torch.nn.functional.softmax(weight, dim=-1)

        # The mean of word vecotrs in a sentence can be a good summarization, why?
        return weight @ x_batch

    def _process_batch(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_indices = self._get_batch(data)
        x_batch = torch.stack([data[i:i+ self._block_size] for i in batch_indices])
        y_batch = torch.stack([data[i+1:i+ self._block_size + 1] for i in batch_indices])
        return x_batch.to(self.device), y_batch.to(self.device)

    def _get_batch(self, data) -> torch.Tensor:
        return torch.randint(len(data) - self._block_size, (self._batch_size,))

    def _update_train_val(self, data: torch.Tensor, train_val_ratio: float) -> None:
        n = int(train_val_ratio * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
