import copy
import pickle
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Union, Callable

import numpy as np
import torch
import torch.nn.functional as F
from snorkel.utils import probs_to_preds
from torch.utils.data import DataLoader

from .dataset import BaseDataset, TorchDataset, BaseSeqDataset
from .evaluation import METRIC, SEQ_METRIC, metric_to_direction


class BaseModel(ABC):
    """Abstract model class."""
    hyperparas: Dict

    @abstractmethod
    def __init__(self, **kwargs: Any):
        pass

    def _update_hyperparas(self, **kwargs: Any):
        for k, v in self.hyperparas.items():
            if k in kwargs: self.hyperparas[k] = kwargs[k]

    @abstractmethod
    def fit(self, dataset_train, y_train=None, dataset_valid=None, y_valid=None,
            verbose: Optional[bool] = False, **kwargs: Any):
        """Abstract method for fitting training data.

        Parameters
        ----------
        """
        pass

    @abstractmethod
    def predict(self, dataset, **kwargs: Any):
        pass

    @abstractmethod
    def test(self, dataset, metric_fn: Union[Callable, str], y_true=None, **kwargs):
        pass

    def save(self, destination: str) -> None:
        """Save label model.
        Parameters
        ----------
        destination
            Filename for saving model
        Example
        -------
        >>> model.save('./saved_model.pkl')  # doctest: +SKIP
        """
        f = open(destination, "wb")
        pickle.dump(self.__dict__, f)
        f.close()

    def load(self, source: str) -> None:
        """Load existing label model.
        Parameters
        ----------
        source
            Filename to load model from
        Example
        -------
        Load parameters saved in ``saved_model``
        >>> model.load('./saved_model.pkl')  # doctest: +SKIP
        """
        f = open(source, "rb")
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)


class BaseCLassModel(BaseModel, ABC):

    @abstractmethod
    def fit(self, dataset_train: Union[BaseDataset, np.ndarray], y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None, y_valid: Optional[np.ndarray] = None,
            verbose: Optional[bool] = False, **kwargs: Any):
        """Abstract method for fitting training data.

        Parameters
        ----------
        """
        pass

    @abstractmethod
    def predict_proba(self, dataset: BaseDataset, **kwargs: Any) -> np.ndarray:
        """Abstract method for outputting probabilistic predictions on given dataset.

        Parameters
        ----------
        """
        pass

    def predict(self, dataset: BaseDataset, **kwargs: Any) -> np.ndarray:
        """Method for predicting on given dataset.

        Parameters
        ----------
        """
        proba = self.predict_proba(dataset, **kwargs)
        return probs_to_preds(probs=proba)

    def test(self, dataset: BaseDataset, metric_fn: Union[Callable, str], y_true: Optional[np.ndarray] = None, **kwargs):
        if isinstance(metric_fn, str):
            metric_fn = METRIC[metric_fn]
        if y_true is None:
            y_true = np.array(dataset.labels)
        probas = self.predict_proba(dataset, **kwargs)
        return metric_fn(y_true, probas)


class BaseSeqModel(BaseModel, ABC):

    @abstractmethod
    def fit(self, dataset_train: BaseSeqDataset, y_train: Optional[List[List]] = None,
            dataset_valid: Optional[BaseSeqDataset] = None, y_valid: Optional[List[List]] = None,
            verbose: Optional[bool] = False, **kwargs: Any):
        """Abstract method for fitting training data.

        Parameters
        ----------
        """
        pass

    @abstractmethod
    def predict(self, dataset: BaseSeqDataset, **kwargs: Any) -> List[List]:
        """Abstract method for predicting on given dataset.

        Parameters
        ----------
        """
        pass

    def test(self, dataset: BaseSeqDataset, metric_fn: Union[Callable, str], y_true: Optional[List[List]] = None,
             strict: Optional[bool] = True, **kwargs):
        if isinstance(metric_fn, str):
            metric_fn = SEQ_METRIC[metric_fn]
        if y_true is None:
            y_true = dataset.labels
        prods = self.predict(dataset, **kwargs)
        return metric_fn(y_true, prods, dataset.id2label, strict)


class BaseTorchClassModel(BaseCLassModel, ABC):
    def _init_valid_dataloader(self, dataset_valid: BaseDataset) -> DataLoader:
        return DataLoader(TorchDataset(dataset_valid), batch_size=self.hyperparas['test_batch_size'], shuffle=False)

    def _init_valid_step(self,
                         dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
                         y_valid: Optional[np.ndarray] = None,
                         metric: Optional[Union[str, Callable]] = 'acc',
                         direction: Optional[str] = 'auto',
                         patience: Optional[int] = 20,
                         tolerance: Optional[float] = -1.0, ):

        if dataset_valid is None:
            self.valid_flag = False
            return False

        if direction == 'auto':
            direction = metric_to_direction(metric)
        assert direction in ['maximize', 'minimize'], f'direction should be in ["maximize", "minimize"]!'
        self.direction = direction

        if isinstance(metric, Callable):
            metric_fn = metric
        else:
            metric_fn = METRIC[metric]
        self.metric_fn = metric_fn

        self.valid_dataloader = self._init_valid_dataloader(dataset_valid)
        if y_valid is None:
            self.y_valid = np.array(dataset_valid.labels)
        else:
            self.y_valid = y_valid
        if direction == 'maximize':
            self.best_metric_value = -np.inf
        else:
            self.best_metric_value = np.inf
        self.best_model = None
        self.best_step = -1
        self.no_improve_cnt = 0

        self.patience = patience
        self.tolerance = tolerance

        self.valid_flag = True
        return True

    def _reset_valid(self):
        if self.direction == 'maximize':
            self.best_metric_value = -np.inf
        else:
            self.best_metric_value = np.inf
        self.best_model = None
        self.best_step = -1
        self.no_improve_cnt = 0

    def _valid_step(self, step, **kwargs):
        early_stop_flag = False
        info = ''
        probas = self.predict_proba(self.valid_dataloader, **kwargs)
        metric_value = self.metric_fn(self.y_valid, probas)
        if (self.direction == 'maximize' and metric_value > self.best_metric_value) or \
                (self.direction == 'minimize' and metric_value < self.best_metric_value):
            self.no_improve_cnt = 0
            self.best_metric_value = metric_value
            self.best_model = copy.deepcopy(self.model.state_dict())
            self.best_step = step
        else:
            self.no_improve_cnt += 1
            if self.patience > 0 and self.no_improve_cnt >= self.patience:
                info = f'[INFO] early stop @ step {step}!'
                early_stop_flag = True
            if self.tolerance > 0 and \
                    ((self.direction == 'maximize' and metric_value + self.tolerance < self.best_metric_value) or
                     (self.direction == 'minimize' and metric_value - self.tolerance < self.best_metric_value)):
                info = f'[INFO] early stop @ step {step} since the gap research {self.tolerance}!'
                early_stop_flag = True
        self.model.train()
        return metric_value, early_stop_flag, info

    def _finalize(self):
        if self.valid_flag:
            self.model.load_state_dict(self.best_model)
            del self.best_model
            del self.valid_dataloader
            del self.y_valid

    def predict_proba(self, dataset: Union[BaseDataset, DataLoader], device: Optional[torch.device] = None, **kwargs: Any):
        if device is not None:
            model = self.model.to(device)
        else:
            model = self.model
        model.eval()
        with torch.no_grad():
            if isinstance(dataset, BaseDataset):
                valid_dataloader = self._init_valid_dataloader(dataset)
            else:
                valid_dataloader = dataset
            probas = []
            for batch in valid_dataloader:
                output = model(batch)
                if output.shape[1] == 1:
                    output = torch.sigmoid(output)
                    proba = torch.cat([1 - output, output], -1)
                else:
                    proba = F.softmax(output, dim=-1)
                probas.append(proba.cpu().detach().numpy())

        return np.vstack(probas)


class BaseTorchSeqModel(BaseSeqModel, ABC):
    @abstractmethod
    def _init_valid_dataloader(self, dataset_valid: BaseSeqDataset) -> DataLoader:
        pass

    def _init_valid_step(self,
                         dataset_valid: Optional[BaseSeqDataset] = None,
                         y_valid: Optional[List[List]] = None,
                         metric: Optional[Union[str, Callable]] = 'f1_seq',
                         strict: Optional[bool] = True,
                         direction: Optional[str] = 'auto',
                         patience: Optional[int] = 20,
                         tolerance: Optional[float] = -1.0, ):

        if dataset_valid is None:
            self.valid_flag = False
            return False

        if direction == 'auto':
            direction = metric_to_direction(metric)
        assert direction in ['maximize', 'minimize'], f'direction should be in ["maximize", "minimize"]!'
        self.direction = direction

        if isinstance(metric, Callable):
            metric_fn = metric
        else:
            metric_fn = partial(SEQ_METRIC[metric], id2label=dataset_valid.id2label, strict=strict)
        self.metric_fn = metric_fn

        self.valid_dataloader = self._init_valid_dataloader(dataset_valid)
        if y_valid is None:
            self.y_valid = dataset_valid.labels
        else:
            self.y_valid = y_valid
        if direction == 'maximize':
            self.best_metric_value = -np.inf
        else:
            self.best_metric_value = np.inf
        self.best_model = None
        self.best_step = -1
        self.no_improve_cnt = 0

        self.patience = patience
        self.tolerance = tolerance

        self.valid_flag = True
        return True

    def _reset_valid(self):
        if self.direction == 'maximize':
            self.best_metric_value = -np.inf
        else:
            self.best_metric_value = np.inf
        self.best_model = None
        self.best_step = -1
        self.no_improve_cnt = 0

    def _valid_step(self, step, **kwargs):
        early_stop_flag = False
        info = ''
        preds = self.predict(self.valid_dataloader, **kwargs)
        metric_value = self.metric_fn(self.y_valid, preds)
        if (self.direction == 'maximize' and metric_value > self.best_metric_value) or \
                (self.direction == 'minimize' and metric_value < self.best_metric_value):
            self.no_improve_cnt = 0
            self.best_metric_value = metric_value
            self.best_model = copy.deepcopy(self.model.state_dict())
            self.best_step = step
        else:
            self.no_improve_cnt += 1
            if self.patience > 0 and self.no_improve_cnt >= self.patience:
                info = f'[INFO] early stop @ step {step}!'
                early_stop_flag = True
            if self.tolerance > 0 and \
                    ((self.direction == 'maximize' and metric_value + self.tolerance < self.best_metric_value) or
                     (self.direction == 'minimize' and metric_value - self.tolerance < self.best_metric_value)):
                info = f'[INFO] early stop @ step {step} since the gap research {self.tolerance}!'
                early_stop_flag = True
        self.model.train()
        return metric_value, early_stop_flag, info

    def _finalize(self):
        if self.valid_flag:
            self.model.load_state_dict(self.best_model)
            del self.best_model
            del self.valid_dataloader
            del self.y_valid

    def predict(self, dataset: Union[BaseSeqDataset, DataLoader], device: Optional[torch.device] = None, **kwargs: Any):
        if device is not None:
            model = self.model.to(device)
        else:
            model = self.model
        model.eval()
        with torch.no_grad():
            if isinstance(dataset, BaseSeqDataset):
                O_id = dataset.label2id['O']
                valid_dataloader = self._init_valid_dataloader(dataset)
            else:
                O_id = dataset.dataset.label2id['O']
                valid_dataloader = dataset
            preds = []
            for batch in valid_dataloader:
                tag_seq = model(batch)
                preds.extend(tag_seq)

        for i, sl in enumerate(valid_dataloader.dataset.seq_len):
            n = len(preds[i])
            if n < sl:
                preds[i].extend([O_id] * (sl - n))
        return preds
