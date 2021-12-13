import logging
from typing import Any, Optional, Union

import numpy as np

from ..basemodel import BaseLabelModel
from ..dataset import BaseDataset
from ..dataset.utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1


class MajorityWeightedVoting(BaseLabelModel):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.balance = None

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            balance: Optional[np.ndarray] = None,
            **kwargs: Any):
        if isinstance(dataset_train, BaseDataset):
            if n_class is not None:
                assert n_class == dataset_train.n_class
            else:
                n_class = dataset_train.n_class
        if n_class is not None and balance is not None:
            assert len(balance) == n_class

        L = check_weak_labels(dataset_train)
        if balance is None:
            self.balance = self._init_balance(L, dataset_valid, y_valid, n_class)
        else:
            self.balance = balance

    def predict_proba(self, dataset: Union[BaseDataset, np.ndarray], **kwargs: Any) -> np.ndarray:
        L = check_weak_labels(dataset)

        n_class = len(self.balance)
        n, m = L.shape
        Y_p = np.zeros((n, n_class))
        for i in range(n):
            counts = np.zeros(n_class)
            for j in range(m):
                if L[i, j] != ABSTAIN:
                    counts[L[i, j]] += self.balance[L[i, j]]
            # Y_p[i, :] = np.where(counts == max(counts), 1, 0)
            if counts.sum() == 0:
                counts += 1
            Y_p[i, :] = counts
        Y_p /= Y_p.sum(axis=1, keepdims=True)
        return Y_p


class MajorityVoting(BaseLabelModel):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.n_class = None

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            n_class: Optional[int] = None,
            **kwargs: Any):
        # warnings.warn(f'MajorityVoting.fit() should not be called!')
        if isinstance(dataset_train, BaseDataset):
            if n_class is not None:
                assert n_class == dataset_train.n_class
            else:
                n_class = dataset_train.n_class
        self.n_class = n_class or int(np.max(check_weak_labels(dataset_train))) + 1

    def predict_proba(self, dataset: Union[BaseDataset, np.ndarray], weight: Optional[np.ndarray] = None,
                      **kwargs: Any) -> np.ndarray:
        L = check_weak_labels(dataset)
        if weight is None:
            weight = np.ones_like(L)

        n_class = self.n_class
        n, m = L.shape
        Y_p = np.zeros((n, n_class))
        for i in range(n):
            counts = np.zeros(n_class)
            for j in range(m):
                if L[i, j] != ABSTAIN:
                    counts[L[i, j]] += 1 * weight[i, j]
            # Y_p[i, :] = np.where(counts == max(counts), 1, 0)
            if counts.sum() == 0:
                counts += 1
            Y_p[i, :] = counts
        Y_p /= Y_p.sum(axis=1, keepdims=True)
        return Y_p
