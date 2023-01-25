#method proposed in paper Learning Hyper Label Model for Programmatic Weak Supervision
# https://arxiv.org/abs/2207.13545
import logging
from typing import Any, Optional, Union

import numpy as np
from hyperlm import HyperLabelModel

from ..basemodel import BaseLabelModel
from ..dataset import BaseDataset
from ..dataset.utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1

class HyperLM(BaseLabelModel):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.n_class = None
        self.hyperlm = HyperLabelModel()

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            n_class: Optional[int] = None,
            **kwargs: Any):
        if isinstance(dataset_train, BaseDataset):
            if n_class is not None:
                assert n_class == dataset_train.n_class
            else:
                n_class = dataset_train.n_class
        self.n_class = n_class or int(np.max(check_weak_labels(dataset_train))) + 1
        self.L_train = check_weak_labels(dataset_train)

    def predict_proba(self, dataset: Union[BaseDataset, np.ndarray],
                      **kwargs: Any) -> np.ndarray:
        L_test = check_weak_labels(dataset)
        if hasattr(self, "L_train"):
            L_all = np.concatenate([self.L_train, L_test])
            Y_p = self.hyperlm.infer(L_all,return_probs=True)
            n_train = self.L_train.shape[0]
            Y_p_test = Y_p[n_train:,:]
        else:
            Y_p_test = self.hyperlm.infer(L_test,return_probs=True)
        return Y_p_test

