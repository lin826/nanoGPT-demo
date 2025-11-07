"""Unit tests for the BatchNorm1d module."""
import torch

from src.train.batch_norm import BatchNorm1d

def test_batch_norm():
    """Tests the BatchNorm1d module for correct mean and standard deviation."""
    torch.manual_seed(1337)  # For reproducibility
    module = BatchNorm1d(100)
    x = torch.randn(32, 100)
    x = module(x)

    print(x[:, 0].mean(), x[:, 0].std())

    assert torch.isclose(x[0, :].mean(), torch.tensor(0.0), atol=1e-7)
    assert torch.isclose(x[0, :].std(), torch.tensor(1.0), atol=1e-9)

if __name__ == "__main__":
    test_batch_norm()
