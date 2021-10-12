import copy
import pickle
from abc import ABC, abstractmethod
from collections import Counter
from functools import partial
from typing import Any, Dict, List, Optional, Union, Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from snorkel.utils import probs_to_preds
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from . import backbone
from .backbone import BackBone, BERTBackBone
from .config import Config
from .dataset import BaseDataset, TorchDataset, BaseSeqDataset
from .evaluation import METRIC, SEQ_METRIC, metric_to_direction
from .utils import get_bert_torch_dataset_class, construct_collate_fn_trunc_pad, get_bert_model_class


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
            verbose: Optional[bool] = False, *args: Any, **kwargs: Any):
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


class BaseTorchModel(BaseModel, ABC):
    model: Optional[BackBone]

    def _init_optimizer_and_lr_scheduler(self,
                                         model: nn.Module,
                                         config: Config,
                                         ) -> Union[Optimizer, Tuple[Optimizer, object]]:
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.optimizer_config['name']
        optimizer_config = config.optimizer_config['paras']
        if optimizer == 'default':
            optimizer_ = AdamW(parameters, lr=optimizer_config['lr'], weight_decay=optimizer_config['weight_decay'])
        else:
            optimizer_ = getattr(optim, optimizer)(parameters, **optimizer_config)
        if hasattr(config, 'lr_scheduler_config'):
            lr_scheduler = config.lr_scheduler_config['name']
            if lr_scheduler == 'default':
                lr_scheduler_ = get_linear_schedule_with_warmup(optimizer_, num_warmup_steps=0, num_training_steps=config.hyperparas['n_steps'])
            else:
                lr_scheduler_config = config.lr_scheduler_config['paras']
                lr_scheduler_ = getattr(optim.lr_scheduler, lr_scheduler)(optimizer_, **lr_scheduler_config)
        else:
            lr_scheduler_ = None
        return optimizer_, lr_scheduler_

    @abstractmethod
    def _init_valid_dataloader(self, dataset_valid, **kwargs: Any) -> DataLoader:
        pass

    @abstractmethod
    def _init_valid_step(self,
                         dataset_valid: Optional = None,
                         y_valid: Optional = None,
                         metric: Optional[Union[str, Callable]] = 'acc',
                         direction: Optional[str] = 'auto',
                         patience: Optional[int] = 20,
                         tolerance: Optional[float] = -1.0,
                         **kwargs: Any):
        pass

    def _init_valid(self,
                    dataset_valid: Optional = None,
                    metric: Optional[Union[str, Callable]] = 'acc',
                    direction: Optional[str] = 'auto',
                    patience: Optional[int] = 20,
                    tolerance: Optional[float] = -1.0,
                    **kwargs: Any,
                    ):

        self.patience = patience
        self.tolerance = tolerance

        if direction == 'auto':
            direction = metric_to_direction(metric)
        assert direction in ['maximize', 'minimize'], f'direction should be in ["maximize", "minimize"]!'
        self.direction = direction

        self.valid_dataloader = self._init_valid_dataloader(dataset_valid, **kwargs)

        self._reset_valid()

    def _reset_valid(self):
        if self.direction == 'maximize':
            self.best_metric_value = -np.inf
        else:
            self.best_metric_value = np.inf
        self.best_model = None
        self.best_step = -1
        self.no_improve_cnt = 0

    @abstractmethod
    def _calc_valid_metric(self, **kwargs):
        pass

    def _valid_step(self, step, **kwargs):
        early_stop_flag = False
        info = ''
        metric_value = self._calc_valid_metric(**kwargs)
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


class BaseClassModel(BaseModel, ABC):

    @abstractmethod
    def fit(self, dataset_train: Union[BaseDataset, np.ndarray], y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None, y_valid: Optional[np.ndarray] = None,
            verbose: Optional[bool] = False, *args: Any, **kwargs: Any):
        """Abstract method for fitting training data.

        Parameters
        ----------
        """
        pass

    @abstractmethod
    def predict_proba(self, dataset: Union[BaseDataset, np.ndarray], **kwargs: Any) -> np.ndarray:
        """Abstract method for outputting probabilistic predictions on given dataset.

        Parameters
        ----------
        """
        pass

    def predict(self, dataset: Union[BaseDataset, np.ndarray], **kwargs: Any) -> np.ndarray:
        """Method for predicting on given dataset.

        Parameters
        ----------
        """
        proba = self.predict_proba(dataset, **kwargs)
        return probs_to_preds(probs=proba)

    def test(self, dataset: Union[BaseDataset, np.ndarray], metric_fn: Union[Callable, str], y_true: Optional[np.ndarray] = None, **kwargs):
        if isinstance(metric_fn, str):
            metric_fn = METRIC[metric_fn]
        if y_true is None:
            y_true = np.array(dataset.labels)
        probas = self.predict_proba(dataset, **kwargs)
        return metric_fn(y_true, probas)


class BaseLabelModel(BaseClassModel):
    """Abstract label model class."""

    @abstractmethod
    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            verbose: Optional[bool] = False,
            *args: Any,
            **kwargs: Any):
        pass

    @staticmethod
    def _init_balance(L: np.ndarray,
                      dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
                      y_valid: Optional[np.ndarray] = None,
                      n_class: Optional[int] = None):
        if y_valid is not None:
            y = y_valid
        elif dataset_valid is not None:
            y = np.array(dataset_valid.labels)
        else:
            y = np.arange(L.max() + 1)
        class_counts = Counter(y)

        if isinstance(dataset_valid, BaseDataset):
            if n_class is not None:
                assert n_class == dataset_valid.n_class
            else:
                n_class = dataset_valid.n_class

        if n_class is None:
            sorted_counts = np.array([v for k, v in sorted(class_counts.items())])
        else:
            sorted_counts = np.zeros(n_class)
            for c, cnt in class_counts.items():
                sorted_counts[c] = cnt
        balance = (sorted_counts + 1) / sum(sorted_counts)

        return balance


class BaseSeqModel(BaseModel, ABC):

    @abstractmethod
    def fit(self, dataset_train: BaseSeqDataset, y_train: Optional[List[List]] = None,
            dataset_valid: Optional[BaseSeqDataset] = None, y_valid: Optional[List[List]] = None,
            verbose: Optional[bool] = False, *args: Any, **kwargs: Any):
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


class BaseTorchClassModel(BaseClassModel, BaseTorchModel, ABC):

    def _init_model(self,
                    dataset: BaseDataset,
                    n_class: int,
                    config: Config,
                    is_bert: Optional[bool] = False,
                    ) -> BackBone:
        hyperparas = config.hyperparas
        if is_bert:
            model = get_bert_model_class(dataset)(
                n_class=n_class,
                binary_mode=hyperparas['binary_mode'],
                **config.backbone_config['paras']
            )
        else:
            input_size = dataset.features.shape[1]
            model = getattr(backbone, config.backbone_config['name'])(
                input_size=input_size,
                n_class=n_class,
                binary_mode=hyperparas['binary_mode'],
                **config.backbone_config['paras']
            )
        return model

    def _init_label_model(self,
                          config: Config,
                          ) -> BaseClassModel:
        from . import labelmodel
        label_model = config.label_model_config['name']
        label_model_config = config.label_model_config['paras']
        label_model_ = getattr(labelmodel, label_model)(**label_model_config)
        return label_model_

    def _init_train_dataloader(self,
                               dataset_train: BaseDataset,
                               n_steps: int,
                               config: Config,
                               return_features: Optional[bool] = False,
                               return_weak_labels: Optional[bool] = False,
                               ) -> DataLoader:
        hyperparas = config.hyperparas
        if isinstance(self.model, BERTBackBone) or (hasattr(self.model, 'backbone') and isinstance(self.model.backbone, BERTBackBone)):
            max_tokens = config.backbone_config['paras']['max_tokens']
            torch_dataset = get_bert_torch_dataset_class(dataset_train)(
                dataset_train,
                self.tokenizer,
                max_tokens,
                n_data=n_steps * hyperparas['batch_size'],
                return_features=return_features,
                return_weak_labels=return_weak_labels,
            )
            train_dataloader = DataLoader(torch_dataset, batch_size=hyperparas['real_batch_size'], shuffle=True, collate_fn=construct_collate_fn_trunc_pad('mask'))
        else:
            torch_dataset = TorchDataset(dataset_train, n_data=n_steps * hyperparas['batch_size'])
            train_dataloader = DataLoader(torch_dataset, batch_size=hyperparas['real_batch_size'], shuffle=True)
        return train_dataloader

    def _init_valid_dataloader(self,
                               dataset_valid: BaseDataset,
                               return_features: Optional[bool] = False,
                               return_weak_labels: Optional[bool] = False,
                               ) -> DataLoader:
        if isinstance(self.model, BERTBackBone) or (hasattr(self.model, 'backbone') and isinstance(self.model.backbone, BERTBackBone)):
            torch_dataset = get_bert_torch_dataset_class(dataset_valid)(
                dataset_valid,
                self.tokenizer,
                512,
                return_features=return_features,
                return_weak_labels=return_weak_labels,
            )
            valid_dataloader = DataLoader(torch_dataset, batch_size=self.hyperparas['test_batch_size'], shuffle=False, collate_fn=construct_collate_fn_trunc_pad('mask'))
            return valid_dataloader
        else:
            return DataLoader(TorchDataset(dataset_valid), batch_size=self.hyperparas['test_batch_size'], shuffle=False)

    def _init_valid_step(self,
                         dataset_valid: Optional[BaseDataset] = None,
                         y_valid: Optional[np.ndarray] = None,
                         metric: Optional[Union[str, Callable]] = 'acc',
                         direction: Optional[str] = 'auto',
                         patience: Optional[int] = 20,
                         tolerance: Optional[float] = -1.0,
                         **kwargs: Any):

        if dataset_valid is None:
            self.valid_flag = False
            return False

        self._init_valid(
            dataset_valid=dataset_valid,
            metric=metric,
            direction=direction,
            patience=patience,
            tolerance=tolerance,
            **kwargs
        )

        if isinstance(metric, Callable):
            metric_fn = metric
        else:
            metric_fn = METRIC[metric]
        self.metric_fn = metric_fn

        if y_valid is None:
            self.y_valid = np.array(dataset_valid.labels)
        else:
            self.y_valid = y_valid

        self.valid_flag = True
        return True

    def _calc_valid_metric(self, **kwargs):
        probas = self.predict_proba(self.valid_dataloader, **kwargs)
        return self.metric_fn(self.y_valid, probas)

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


class BaseTorchSeqModel(BaseSeqModel, BaseTorchModel, ABC):

    def _init_label_model(self,
                          config: Config,
                          ) -> BaseSeqModel:
        from . import labelmodel, seq_labelmodel
        label_model = config.label_model_config['name']
        label_model_config = config.label_model_config['paras']
        if hasattr(seq_labelmodel, label_model):
            label_model_ = getattr(seq_labelmodel, label_model)(**label_model_config)
        else:
            from .seq_labelmodel import SeqLabelModelWrapper
            label_model_ = SeqLabelModelWrapper(label_model_class=getattr(labelmodel, label_model), **label_model_config)
        return label_model_

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

        self._init_valid(
            dataset_valid=dataset_valid,
            metric=metric,
            direction=direction,
            patience=patience,
            tolerance=tolerance,
        )

        if isinstance(metric, Callable):
            metric_fn = metric
        else:
            metric_fn = partial(SEQ_METRIC[metric], id2label=dataset_valid.id2label, strict=strict)
        self.metric_fn = metric_fn

        if y_valid is None:
            self.y_valid = dataset_valid.labels
        else:
            self.y_valid = y_valid

        self.valid_flag = True
        return True

    def _calc_valid_metric(self, **kwargs):
        preds = self.predict(self.valid_dataloader, **kwargs)
        return self.metric_fn(self.y_valid, preds)

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
