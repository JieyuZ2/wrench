import copy
import json
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Union, Callable

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from snorkel.labeling import LFAnalysis
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """Abstract data class."""

    def __init__(self,
                 path: str = None,
                 split: Optional[str] = None,
                 feature_cache_name: Optional[str] = None,
                 **kwargs: Any) -> None:
        self.ids: List = []
        self.labels: List = []
        self.examples: List = []
        self.weak_labels: List[List] = []
        self.features = None
        self.id2label = None

        self.split = split
        self.path = path

        if path is not None and split is not None:
            self.load(path=path, split=split)
            self.load_features(feature_cache_name)
            self.n_class = len(self.id2label)
            self.n_lf = len(self.weak_labels[0])

    def __len__(self):
        return len(self.ids)

    def load(self, path: str, split: str):
        """Method for loading data given the split.

        Parameters
        ----------
        split
            A str with values in {"train", "valid", "test", None}. If None, then do not load any data.
        Returns
        -------
        self
        """

        assert split in ["train", "valid", "test"], 'Parameter "split" must be in ["train", "valid", "test", None]'

        path = Path(path)

        self.split = split
        self.path = path

        data_path = path / f'{split}.json'
        logger.info(f'loading data from {data_path}')
        data = json.load(open(data_path, 'r'))
        for i, item in tqdm(data.items()):
            self.ids.append(i)
            self.labels.append(item['label'])
            self.weak_labels.append(item['weak_labels'])
            self.examples.append(item['data'])

        label_path = self.path / f'label.json'
        self.id2label = {int(k): v for k, v in json.load(open(label_path, 'r')).items()}

        return self

    def load_labeled_ids_and_lf_exemplars(self, path: str):

        path = Path(path)

        assert self.split == 'train', 'labeled data can only be loaded by train'
        logger.info(f'loading labeled ids and lf exemplars from {path}')
        data = json.load(open(path, 'r'))
        labeled_ids = data.get('labeled_ids', [])
        lf_exemplar_ids = data.get('lf_exemplar_ids', [])

        # map to real data idx in self
        labeled_ids = [self.ids.index(i) for i in labeled_ids]
        lf_exemplar_ids = [self.ids.index(i) for i in lf_exemplar_ids]

        return labeled_ids, lf_exemplar_ids

    def load_features(self, cache_name: Optional[str] = None):
        """Method for loading data feature given the split and cache_name.

        Parameters
        ----------
        cache_name
            A str used to locate the feature file.
        Returns
        -------
        features
            np.ndarray
        """
        if cache_name is None:
            self.features = None
            return None

        path = self.path / f'{self.split}_{cache_name}.pkl'
        logger.info(f'loading features from {path}')
        features = pickle.load(open(path, 'rb'))
        self.features = features
        return features

    def save_features(self, cache_name: Optional[str] = None):
        if cache_name is None:
            return None
        path = self.path / f'{self.split}_{cache_name}.pkl'
        logger.info(f'saving features into {path}')
        pickle.dump(self.features, open(path, 'wb'), protocol=4)
        return path

    def extract_feature(self,
                        extract_fn: Union[str, Callable],
                        return_extractor: bool,
                        cache_name: str = None,
                        force: bool = False,
                        normalize=False,
                        **kwargs: Any):
        if cache_name is not None:
            path = self.path / f'{self.split}_{cache_name}.pkl'
            if path.exists() and (not force):
                self.load_features(cache_name=cache_name)
                return

        if isinstance(extract_fn, Callable):
            self.features = extract_fn(self.examples)
        else:
            extractor = self.extract_feature_(extract_fn=extract_fn, return_extractor=return_extractor, **kwargs)
            if normalize:
                features = self.features
                scaler = preprocessing.StandardScaler().fit(features)
                self.features = scaler.transform(features)
                extract_fn = lambda x: scaler.transform(extractor(x))
            else:
                extract_fn = extractor

        if cache_name is not None:
            self.save_features(cache_name=cache_name)

        if return_extractor:
            return extract_fn

    @abstractmethod
    def extract_feature_(self, extract_fn: str, return_extractor: bool, **kwargs: Any):
        """Abstract method for extracting features given the mode.

        Parameters
        ----------
        """
        pass

    def create_subset(self, idx: List[int]):
        dataset = self.__class__()
        for i in idx:
            dataset.ids.append(self.ids[i])
            dataset.labels.append(self.labels[i])
            dataset.examples.append(self.examples[i])
            dataset.weak_labels.append(self.weak_labels[i])

        if self.features is not None:
            dataset.features = self.features[idx]

        dataset.id2label = copy.deepcopy(self.id2label)
        dataset.split = self.split
        dataset.path = self.path
        dataset.n_class = self.n_class
        dataset.n_lf = self.n_lf

        return dataset

    def create_split(self, idx: List[int]):
        chosen = self.create_subset(idx)
        remain = self.create_subset([i for i in range(len(self)) if i not in idx])
        return chosen, remain

    def sample(self, alpha: Union[int, float], return_dataset=True):
        if isinstance(alpha, float):
            alpha = int(len(self) * alpha)
        idx = np.random.choice(len(self), alpha, replace=False)
        if return_dataset:
            return self.create_subset(idx)
        else:
            return list(idx)

    def get_covered_subset(self):
        idx = [i for i in range(len(self)) if np.any(np.array(self.weak_labels[i]) != -1)]
        return self.create_subset(idx)

    def get_conflict_labeled_subset(self):
        idx = [i for i in range(len(self)) if len({l for l in set(self.weak_labels[i]) if l != -1}) > 1]
        return self.create_subset(idx)

    def get_agreed_labeled_subset(self):
        idx = [i for i in range(len(self)) if len({l for l in set(self.weak_labels[i]) if l != -1}) == 1]
        return self.create_subset(idx)

    def lf_summary(self):
        L = np.array(self.weak_labels)
        Y = np.array(self.labels)
        lf_summary = LFAnalysis(L=L).lf_summary(Y=Y)
        return lf_summary

    def summary(self, n_clusters=10, features=None, return_lf_summary=False):
        summary_d = {}
        L = np.array(self.weak_labels)
        Y = np.array(self.labels)

        summary_d['n_class'] = self.n_class
        summary_d['n_data'], summary_d['n_lfs'] = L.shape
        summary_d['n_uncovered_data'] = np.sum(np.all(L == -1, axis=1))
        uncovered_rate = summary_d['n_uncovered_data'] / summary_d['n_data']
        summary_d['overall_coverage'] = (1 - uncovered_rate)

        lf_summary = LFAnalysis(L=L).lf_summary(Y=Y)
        summary_d['lf_avr_acc'] = lf_summary['Emp. Acc.'].mean()
        summary_d['lf_var_acc'] = lf_summary['Emp. Acc.'].var()
        summary_d['lf_avr_propensity'] = lf_summary['Coverage'].mean()
        summary_d['lf_var_propensity'] = lf_summary['Coverage'].var()
        summary_d['lf_avr_overlap'] = lf_summary['Overlaps'].mean()
        summary_d['lf_var_overlap'] = lf_summary['Overlaps'].var()
        summary_d['lf_avr_conflict'] = lf_summary['Conflicts'].mean()
        summary_d['lf_var_conflict'] = lf_summary['Conflicts'].var()

        # calc cmi
        from ..utils import calc_cmi_matrix, cluster_based_accuracy_variance
        cmi_matrix = calc_cmi_matrix(Y, L)
        lf_cmi = np.ma.masked_invalid(cmi_matrix).mean(1).data
        summary_d['correlation'] = lf_cmi.mean()
        lf_summary['correlation'] = pd.Series(lf_cmi)

        # calc data dependency
        if hasattr(self, 'features') and features is None:
            features = self.features
        if features is not None:
            kmeans = KMeans(n_clusters=n_clusters).fit(features)
            cluster_labels = kmeans.labels_
            acc_var = np.array([cluster_based_accuracy_variance(Y, L[:, i], cluster_labels) for i in range(self.n_lf)])
            summary_d['data-dependency'] = acc_var.mean()
            lf_summary['data-dependency'] = pd.Series(acc_var)

        if return_lf_summary:
            return summary_d, lf_summary
        else:
            return summary_d
