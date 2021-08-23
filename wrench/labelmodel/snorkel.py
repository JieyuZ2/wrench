from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import warnings
import numpy as np

from snorkel.labeling.model import LabelModel

from .baselabelmodel import BaseLabelModel
from ..dataset import BaseDataset
from .utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1


class Snorkel(BaseLabelModel):
    def __init__(self,
                 lr: Optional[float] = 0.01,
                 l2: Optional[float] = 0.0,
                 n_epochs: Optional[int] = 100,
                 **kwargs: Any):
        super().__init__()
        self.hyperparas = {
            'lr': lr,
            'l2': l2,
            'n_epochs': n_epochs,
        }
        self.model = None

    def fit(self,
            dataset_train:Union[BaseDataset, np.ndarray],
            y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            balance: Optional[np.ndarray] = None,
            verbose: Optional[bool] = False,
            seed: int =None, **kwargs: Any):

        self._update_hyperparas(**kwargs)

        L = check_weak_labels(dataset_train)
        balance = balance or self._init_balance(L, dataset_valid, y_valid)
        seed = seed or np.random.randint(1e6)

        label_model = LabelModel(cardinality=len(balance), verbose=verbose)
        label_model.fit(L_train=L, class_balance=balance, n_epochs=self.hyperparas['n_epochs'],
                        lr=self.hyperparas['lr'], l2=self.hyperparas['l2'], seed=seed)

        self.model = label_model

    def predict_proba(self, dataset:Union[BaseDataset, np.ndarray], **kwargs: Any) -> np.ndarray:
        L = check_weak_labels(dataset)
        return self.model.predict_proba(L)