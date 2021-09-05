from abc import abstractmethod
from collections import Counter
from typing import Any, Optional, Union

import numpy as np

from ..basemodel import BaseCLassModel
from ..dataset import BaseDataset


def check_weak_labels(dataset: Union[BaseDataset, np.ndarray]) -> np.ndarray:
    if isinstance(dataset, BaseDataset):
        assert dataset.weak_labels is not None, f'Input dataset has no weak labels!'
        L = np.array(dataset.weak_labels)
    else:
        L = dataset
    return L


class BaseLabelModel(BaseCLassModel):
    """Abstract label model class."""

    @abstractmethod
    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            verbose: Optional[bool] = False,
            **kwargs: Any):
        pass

    @staticmethod
    def _init_balance(L: np.ndarray,
                      dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
                      y_valid: Optional[np.ndarray] = None,
                      n_class: Optional[int] = None):
        if y_valid is not None:
            y = y_valid
        elif dataset_valid is not None:
            y = np.array(dataset_valid.labels)
        else:
            y = np.arange(L.max() + 1)
        class_counts = Counter(y)

        if isinstance(dataset_valid, BaseDataset):
            if n_class is not None:
                assert n_class == dataset_valid.n_class
            else:
                n_class = dataset_valid.n_class

        if n_class is None:
            sorted_counts = np.array([v for k, v in sorted(class_counts.items())])
        else:
            sorted_counts = np.zeros(n_class)
            for c, cnt in class_counts.items():
                sorted_counts[c] = cnt
        balance = (sorted_counts + 1) / sum(sorted_counts)

        return balance
