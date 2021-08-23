from typing import Optional
import math
import numpy as np
from torch.utils.data import Dataset

from ..dataset import BaseDataset


def sample_batch(loader):
    while True:
        for batch in loader:
            yield batch


class TorchDataset(Dataset):
    def __init__(self, dataset: BaseDataset, n_data: Optional[int] = 0):
        self.features = dataset.features
        self.labels = dataset.labels
        self.weak_labels = np.array(dataset.weak_labels, dtype=np.float32)
        self.data = dataset.examples
        n_data_ = len(self.data)
        if n_data > 0:
            self.n_data = math.ceil(n_data / n_data_) * n_data_
        else:
            self.n_data = n_data_

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        d = {
            'ids': idx,
            'labels': self.labels[idx],
            'weak_labels': self.weak_labels[idx],
            'data': self.data[idx],
        }
        if self.features is not None:
            d['features'] = self.features[idx]
        return d