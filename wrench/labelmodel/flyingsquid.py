from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import numpy as np

from flyingsquid.label_model import LabelModel

from .baselabelmodel import BaseLabelModel
from ..dataset import BaseDataset
from .utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1


class FlyingSquid(BaseLabelModel):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.hyperparas = {}
        self.model = None
        self.cardinality = None

    def fit(self,
            dataset_train:Union[BaseDataset, np.ndarray],
            y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            balance: Optional[np.ndarray] = None,
            dependency_graph: Optional[List] = [],
            verbose: Optional[bool] = False,
            **kwargs: Any):

        self._update_hyperparas(**kwargs)

        L = check_weak_labels(dataset_train)
        balance = balance or self._init_balance(L, dataset_valid, y_valid)
        cardinality = len(balance)
        self.cardinality = cardinality

        n, m = L.shape
        if cardinality > 2:
            model = []
            for i in range(cardinality):
                label_model = LabelModel(m=m, lambda_edges=dependency_graph)
                L_i = np.copy(L)
                target_mask = L_i == i
                abstain_mask = L_i == ABSTAIN
                other_mask = (~target_mask) & (~abstain_mask)
                L_i[target_mask] = 1
                L_i[abstain_mask] = 0
                L_i[other_mask] = -1
                label_model.fit(L_train=L_i, class_balance=np.array([1-balance[i], balance[i]]), verbose=verbose, **kwargs)
                model.append(label_model)
        else:
            model = LabelModel(m=m, lambda_edges=dependency_graph)
            L_i = np.copy(L)
            abstain_mask = L_i == -1
            negative_mask = L_i == 0
            L_i[abstain_mask] = 0
            L_i[negative_mask] = -1
            model.fit(L_train=L_i, class_balance=balance, verbose=verbose, **kwargs)

        self.model = model

    def predict_proba(self, dataset:Union[BaseDataset, np.ndarray], **kwargs: Any) -> np.ndarray:
        L = check_weak_labels(dataset)
        if self.cardinality > 2:
            probas = np.zeros((len(L), self.cardinality))
            for i in range(self.cardinality):
                L_i = np.copy(L)
                target_mask = L_i == i
                abstain_mask = L_i == ABSTAIN
                other_mask = (~target_mask) & (~abstain_mask)
                L_i[target_mask] = 1
                L_i[abstain_mask] = 0
                L_i[other_mask] = -1
                probas[:, i] = self.model[i].predict_proba(L_matrix=L_i)[:, 1]
            probas = np.nan_to_num(probas, nan=-np.inf) # handle NaN
            probas = np.exp(probas) / np.sum(np.exp(probas), axis=1, keepdims=True)
        else:
            L_i = np.copy(L)
            abstain_mask = L_i == -1
            negative_mask = L_i == 0
            L_i[abstain_mask] = 0
            L_i[negative_mask] = -1
            probas = self.model.predict_proba(L_matrix=L_i)
        return probas