from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
from collections import Counter
import numpy as np
from snorkel.labeling import LFAnalysis

from .baselabelmodel import BaseLabelModel
from ..dataset import BaseDataset
from .utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1


class GoldCondProb(BaseLabelModel):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.cond_probs = None
        self.cardinality = None
        self.balance = None

    def fit(self,
            dataset_train: BaseDataset,
            dataset_valid: Optional[BaseDataset] = None,
            use_prior=True,
            **kwargs: Any):
        if np.all(np.array(dataset_train.labels)==-1):
            dataset_train = dataset_valid
        L = check_weak_labels(dataset_train)
        Y = np.array(dataset_train.labels)
        class_counts = Counter(Y)
        sorted_counts = np.array([v for k, v in sorted(class_counts.items())])
        self.cardinality = len(sorted_counts)
        if use_prior:
            self.balance = sorted_counts / sum(sorted_counts)
        else:
            self.balance = np.ones(self.cardinality)
        self.cond_probs = LFAnalysis(L).lf_empirical_probs(Y, self.cardinality)

    def predict_proba(self, dataset:Union[BaseDataset, np.ndarray], weight: Optional[np.ndarray] = None,
                      **kwargs: Any) -> np.ndarray:
        L = check_weak_labels(dataset)
        n, m = L.shape
        L_shift = L + 1  # convert to {0, 1, ..., k}
        L_aug = np.zeros((n, m * self.cardinality))
        for y in range(1, self.cardinality + 1):
            # A[x::y] slices A starting at x at intervals of y
            # e.g., np.arange(9)[0::3] == np.array([0,3,6])
            L_aug[:, (y - 1):: self.cardinality] = np.where(L_shift == y, 1, 0)
        mu = self.cond_probs[:, 1:, :].reshape(-1, self.cardinality)
        mu_eps = min(0.01, 1 / 10 ** np.ceil(np.log10(n)))
        mu = np.clip(mu, mu_eps, 1 - mu_eps)
        jtm = np.ones(L_aug.shape[1])
        # Note: We omit abstains, effectively assuming uniform distribution here
        X = np.exp(L_aug @ np.diag(jtm) @ np.log(mu) + np.log(self.balance))
        Z = np.tile(X.sum(axis=1).reshape(-1, 1), self.cardinality)
        return X / Z