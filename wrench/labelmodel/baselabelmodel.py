from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import Counter
import numpy as np

from ..basemodel import BaseModel
from ..dataset import BaseDataset



class BaseLabelModel(BaseModel):
    """Abstract label model class."""

    @staticmethod
    def _init_balance(L: np.ndarray,
                      dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
                      y_valid: Optional[np.ndarray] = None, ):
        if y_valid is not None:
            y = y_valid
        elif dataset_valid is not None:
            y = np.array(dataset_valid.labels)
        else:
            y = np.arange(L.max()+1)
        class_counts = Counter(y)
        sorted_counts = np.array([v for k, v in sorted(class_counts.items())])
        balance = sorted_counts / sum(sorted_counts)
        return balance