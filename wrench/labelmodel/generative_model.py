from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import warnings
from collections import Counter
import numpy as np

from .generative_model_src import SrcGenerativeModel

from .baselabelmodel import BaseLabelModel
from ..dataset import BaseDataset
from .utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1


class GenerativeModel(BaseLabelModel):
    def __init__(self,
                 lr: Optional[float] = 1e-4,
                 l2: Optional[float] = 1e-1,
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
            threads: Optional[int] = 10,
            verbose: Optional[bool] = False,
            seed: int =None,
            **kwargs: Any):

        self._update_hyperparas(**kwargs)

        L = check_weak_labels(dataset_train)
        balance = balance or self._init_balance(L, dataset_valid, y_valid)
        cardinality = len(balance)
        self.cardinality = cardinality
        L = self.process_label_matrix(L)

        seed = seed or np.random.randint(1e6)

        ## TODO support multiclass class prior
        log_y_prior = np.log(balance)
        label_model = SrcGenerativeModel(cardinality=cardinality, class_prior=False, seed=seed)
        label_model.train(L=L, init_class_prior=log_y_prior, epochs=self.hyperparas['n_epochs'],
                          step_size=self.hyperparas['lr'], reg_param=self.hyperparas['l2'],
                          verbose=verbose, cardinality=cardinality, threads=threads)

        self.model = label_model

    def predict_proba(self, dataset:Union[BaseDataset, np.ndarray], **kwargs: Any) -> np.ndarray:
        L = check_weak_labels(dataset)
        L = self.process_label_matrix(L)
        return self.model.predict_proba(L)

    def process_label_matrix(self, L_):
        L = L_.copy()
        if self.cardinality > 2:
            L += 1
        else:
            abstain_mask = L == -1
            negative_mask = L == 0
            L[abstain_mask] = 0
            L[negative_mask] = -1
        return L
