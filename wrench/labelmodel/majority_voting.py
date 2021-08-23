from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import numpy as np

from .baselabelmodel import BaseLabelModel
from ..dataset import BaseDataset
from .utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1


class MajorityWeightedVoting(BaseLabelModel):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.balance = None

    def fit(self, dataset_train:Union[BaseDataset, np.ndarray], y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None, y_valid: Optional[np.ndarray] = None,
            balance: Optional[np.ndarray] = None, **kwargs: Any):
        L = check_weak_labels(dataset_train)
        self.balance = balance or self._init_balance(L, dataset_valid, y_valid)

    def predict_proba(self, dataset:Union[BaseDataset, np.ndarray], **kwargs: Any) -> np.ndarray:
        L = check_weak_labels(dataset)

        cardinality = len(self.balance)
        n, m = L.shape
        Y_p = np.zeros((n, cardinality))
        for i in range(n):
            counts = np.zeros(cardinality)
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
        self.cardinality = None

    def fit(self, dataset_train:Union[BaseDataset, np.ndarray], cardinality: Optional[int] = None, **kwargs: Any):
        # warnings.warn(f'MajorityVoting.fit() should not be called!')
        L = check_weak_labels(dataset_train)
        self.cardinality = cardinality or np.max(L) + 1

    def predict_proba(self, dataset:Union[BaseDataset, np.ndarray], weight: Optional[np.ndarray] = None,
                      **kwargs: Any) -> np.ndarray:
        L = check_weak_labels(dataset)
        if weight is None:
            weight = np.ones_like(L)

        cardinality = self.cardinality
        n, m = L.shape
        Y_p = np.zeros((n, cardinality))
        for i in range(n):
            counts = np.zeros(cardinality)
            for j in range(m):
                if L[i, j] != ABSTAIN:
                    counts[L[i, j]] += 1 * weight[i, j]
            # Y_p[i, :] = np.where(counts == max(counts), 1, 0)
            if counts.sum() == 0:
                counts += 1
            Y_p[i, :] = counts
        Y_p /= Y_p.sum(axis=1, keepdims=True)
        return Y_p