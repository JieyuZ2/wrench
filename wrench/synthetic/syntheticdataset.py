import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.utils import check_random_state

from ..dataset.basedataset import BaseDataset


class BaseSyntheticGenerator(ABC):
    def __init__(self, n_class: int, n_lfs: int, class_prior: Optional[Union[list, np.ndarray]] = None,
                 lf_prior: Optional[Union[list, np.ndarray]] = None, random_state=None):
        self.generator = check_random_state(random_state)

        self.n_class = n_class
        self.n_lfs = n_lfs
        if class_prior is None:
            class_prior = np.ones(n_class) / n_class
        self.class_prior = np.array(class_prior)
        if lf_prior is None:
            lf_prior = np.ones(n_class) / n_class
        self.lf_prior = np.array(lf_prior)
        self.lf_targets = self.generator.choice(n_class, size=n_lfs, p=lf_prior)

        self.id2labels = {i: i for i in range(n_class)}

    def generate_split(self, split: str = 'train', n_data: int = 1000):
        generated = self.generate(n_data=n_data)
        dataset = SyntheticDataset(split=split, id2label=self.id2labels.copy(), **generated)
        return dataset

    def sample_other_label(self, label):
        other_labels = [i for i in range(self.n_class) if i != label]
        p = np.array([self.class_prior[i] for i in other_labels])
        p = p / np.sum(p)
        return self.generator.choice(other_labels, p=p)

    @abstractmethod
    def generate(self, n_data: int = 1000):
        pass


class SyntheticDataset(BaseDataset):
    """Data class for synthetic dataset."""

    def __init__(self,
                 split: str,
                 ids: List,
                 labels: List,
                 examples: List,
                 weak_labels: List[List],
                 id2label: Dict,
                 features: Optional[np.ndarray] = None,
                 **kwargs: Any) -> None:
        self.ids = ids
        self.labels = labels
        self.examples = examples
        self.weak_labels = weak_labels
        self.features = features

        self.path = None
        self.split = split
        self.id2label = id2label

        self.n_class = len(self.id2label)
        self.n_lf = len(self.weak_labels[0])

    def sample(self, alpha: Union[int, float]):
        if isinstance(alpha, float):
            alpha = int(len(self) * alpha)
        idx = np.random.choice(len(self), alpha, replace=False)
        ids, labels, examples, weak_labels = [], [], [], []
        for i in idx:
            ids.append(self.ids[i])
            labels.append(self.labels[i])
            examples.append(self.examples[i])
            weak_labels.append(self.weak_labels[i])
        if self.features is not None:
            features = self.features[idx]
        else:
            features = None

        dataset = self.__class__(split=self.split, id2label=self.id2label.copy(),
                                 ids=ids, labels=labels, examples=examples, weak_labels=weak_labels, features=features)

        return dataset

    def extract_feature_(self, **kwargs: Any):
        warnings.warn(f'synthetic dataset have no feature!')
