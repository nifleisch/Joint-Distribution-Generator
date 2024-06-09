import torch
from torch import Tensor
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    def __init__(self, n_samples: int, d: int) -> None:
        self.n_samples = n_samples
        self.d = d

    def __len__(self) -> int:
        return 4096

    def __getitem__(self, idx) -> Tensor:
        return torch.rand((self.n_samples, self.d))

    def sample(self, n_samples: int) -> Tensor:
        return torch.rand((n_samples, self.d))
