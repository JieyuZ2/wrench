import logging
from typing import Any, Optional, Union

import numpy as np

from .generative_model_src import SrcGenerativeModel
from ..basemodel import BaseLabelModel
from ..dataset import BaseDataset
from ..dataset.utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1


class GenerativeModel(BaseLabelModel):
    def __init__(self,
                 lr: Optional[float] = 1e-4,
                 l2: Optional[float] = 1e-1,
                 n_epochs: Optional[int] = 100,
                 seed: Optional[int] = None,
                 **kwargs: Any):
        super().__init__()
        self.hyperparas = {
            'lr'      : lr,
            'l2'      : l2,
            'n_epochs': n_epochs,
            'seed'    : seed or np.random.randint(1e6),
        }
        self.model = None

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            balance: Optional[np.ndarray] = None,
            threads: Optional[int] = 10,
            verbose: Optional[bool] = False,
            **kwargs: Any):

        self._update_hyperparas(**kwargs)
        if isinstance(dataset_train, BaseDataset):
            if n_class is not None:
                assert n_class == dataset_train.n_class
            else:
                n_class = dataset_train.n_class
        if n_class is not None and balance is not None:
            assert len(balance) == n_class

        L = check_weak_labels(dataset_train)
        if balance is None:
            balance = self._init_balance(L, dataset_valid, y_valid, n_class)
        n_class = len(balance)
        self.n_class = n_class

        L = self.process_label_matrix(L)

        ## TODO support multiclass class prior
        log_y_prior = np.log(balance)
        label_model = SrcGenerativeModel(cardinality=n_class, class_prior=False, seed=self.hyperparas['seed'])
        label_model.train(
            L=L,
            init_class_prior=log_y_prior,
            epochs=self.hyperparas['n_epochs'],
            step_size=self.hyperparas['lr'],
            reg_param=self.hyperparas['l2'],
            verbose=verbose,
            cardinality=n_class,
            threads=threads)

        self.model = label_model

    def predict_proba(self, dataset: Union[BaseDataset, np.ndarray], **kwargs: Any) -> np.ndarray:
        L = check_weak_labels(dataset)
        L = self.process_label_matrix(L)
        return self.model.predict_proba(L)

    def process_label_matrix(self, L_):
        L = L_.copy()
        if self.n_class > 2:
            L += 1
        else:
            abstain_mask = L == -1
            negative_mask = L == 0
            L[abstain_mask] = 0
            L[negative_mask] = -1
        return L
