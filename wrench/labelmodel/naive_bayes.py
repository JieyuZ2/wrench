from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import numpy as np

from .label_model_src import NaiveBayes, LearningConfig

from .baselabelmodel import BaseLabelModel
from ..dataset import BaseDataset
from .utils import check_weak_labels

logger = logging.getLogger(__name__)


class NaiveBayesModel(BaseLabelModel):
    def __init__(self,
                 lr: Optional[float] = 0.01,
                 reg_weight: Optional[float] = 1e-1,
                 momentum: Optional[float] = 0.9,
                 n_epochs: Optional[int] = 5,
                 batch_size: Optional[int] = 64,
                 **kwargs: Any):
        super().__init__()
        self.hyperparas = {
            'lr': lr,
            'momentum': momentum,
            'reg_weight': reg_weight,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
        }
        self.model = None

    def fit(self,
            dataset_train:Union[BaseDataset, np.ndarray],
            y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            verbose: Optional[bool] = False,
            seed: int =None,
            **kwargs: Any):

        self._update_hyperparas(**kwargs)
        config = LearningConfig()
        config.epochs = self.hyperparas['n_epochs']
        config.batch_size = self.hyperparas['batch_size']
        config.step_size = self.hyperparas['lr']
        config.momentum = self.hyperparas['momentum']
        config.seed = seed or np.random.randint(1e6)

        L_shift = check_weak_labels(dataset_train) + 1
        self.cardinality = L_shift.max()

        label_model = NaiveBayes(num_classes=self.cardinality, num_lfs=L_shift.shape[1])
        label_model.estimate_label_model(votes=L_shift, config=config)

        self.model = label_model

    def predict_proba(self, dataset:Union[BaseDataset, np.ndarray], **kwargs: Any) -> np.ndarray:
        L_shift = check_weak_labels(dataset) + 1
        return self.model.get_label_distribution(L_shift)
