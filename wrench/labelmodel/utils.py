from typing import Union
import numpy as np
from ..dataset import BaseDataset

def check_weak_labels(dataset:Union[BaseDataset, np.ndarray]) -> np.ndarray:
    if isinstance(dataset, BaseDataset):
        assert dataset.weak_labels is not None, f'Input dataset has no weak labels!'
        L = np.array(dataset.weak_labels)
    else:
        L = dataset
    return L


