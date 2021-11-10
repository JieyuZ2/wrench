import logging
from collections import Counter
from typing import Any, Optional, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange
from transformers import AutoTokenizer

from ..backbone import BackBone
from ..basemodel import BaseTorchClassModel, BaseLabelModel
from ..config import Config
from ..dataset import BaseDataset
from ..dataset.utils import split_labeled_unlabeled
from ..utils import cross_entropy_with_probs

logger = logging.getLogger(__name__)

ABSTAIN = -1


class Encoder(BackBone):
    def __init__(self, input_size, n_rules, hidden_size, n_class, temperature, dropout=0.3, balance=None):
        super(Encoder, self).__init__(n_class=n_class)
        self.n_rules = n_rules
        self.acc_scaler = np.sqrt(n_rules) * n_class
        self.temperature = temperature
        self.encoder = nn.Sequential(
            nn.Linear(input_size + n_rules, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_rules * n_class),
        )
        if balance is None:
            self.log_class_prior = nn.Parameter(torch.log(torch.ones(n_class) / n_class), requires_grad=False)
        else:
            self.log_class_prior = nn.Parameter(torch.from_numpy(balance), requires_grad=False)

    def set_class_prior(self, p):
        for i, pp in enumerate(p):
            self.log_class_prior.data[i] = np.log(p[i])

    def forward(self, batch):
        device = self.get_device()
        weak_labels = batch['weak_labels'].to(device)
        features = batch['features'].to(device)
        batch_size = features.size(0)

        x = torch.cat((weak_labels, features), 1)
        z = self.encoder(x).view(batch_size, self.n_rules, self.n_class) / self.temperature

        mask = weak_labels != ABSTAIN
        z = self.acc_scaler * torch.softmax(z, dim=1) * torch.unsqueeze(mask, dim=2)

        one_hot = F.one_hot(weak_labels.long() * mask, num_classes=self.n_class)
        z = z * one_hot

        logits = torch.sum(z, dim=1) + self.log_class_prior

        return logits


class WeaSELModel(BackBone):
    def __init__(self, input_size, n_rules, hidden_size, n_class, temperature, dropout, backbone, balance):
        super(WeaSELModel, self).__init__(n_class=n_class)
        self.backbone = backbone
        self.encoder = Encoder(input_size, n_rules, hidden_size, n_class, temperature, dropout, balance)
        self.forward = self.backbone.forward

    def calculate_loss(self, batch):
        predict_f = self.backbone(batch)
        predict_e = self.encoder(batch)
        loss_f = cross_entropy_with_probs(predict_f, torch.softmax(predict_e.detach(), dim=-1))
        loss_e = cross_entropy_with_probs(predict_e, torch.softmax(predict_f.detach(), dim=-1))
        loss = loss_e + loss_f
        return loss


def init_balance(n_class: int,
                 dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
                 y_valid: Optional[np.ndarray] = None
                 ):
    if y_valid is not None:
        y = y_valid
    elif dataset_valid is not None:
        y = np.array(dataset_valid.labels)
    else:
        return np.ones(n_class) / n_class
    class_counts = Counter(y)

    if isinstance(dataset_valid, BaseDataset):
        assert n_class == dataset_valid.n_class

    sorted_counts = np.zeros(n_class)
    for c, cnt in class_counts.items():
        sorted_counts[c] = cnt
    balance = (sorted_counts + 1) / sum(sorted_counts)

    return balance


class WeaSEL(BaseTorchClassModel):
    def __init__(self,
                 temperature: Optional[float] = 1.0,
                 dropout: Optional[float] = 0.3,
                 hidden_size: Optional[int] = 100,

                 batch_size: Optional[int] = 16,
                 real_batch_size: Optional[int] = 16,
                 test_batch_size: Optional[int] = 16,
                 n_steps: Optional[int] = 10000,
                 grad_norm: Optional[float] = -1,
                 use_lr_scheduler: Optional[bool] = False,
                 binary_mode: Optional[bool] = False,
                 **kwargs: Any
                 ):
        super().__init__()
        self.hyperparas = {
            'temperature'     : temperature,
            'dropout'         : dropout,
            'hidden_size'     : hidden_size,

            'batch_size'      : batch_size,
            'real_batch_size' : real_batch_size,
            'test_batch_size' : test_batch_size,
            'n_steps'         : n_steps,
            'grad_norm'       : grad_norm,
            'use_lr_scheduler': use_lr_scheduler,
            'binary_mode'     : binary_mode,
        }
        self.model: Optional[WeaSELModel] = None
        self.label_model: Optional[BaseLabelModel] = None
        self.config = Config(
            self.hyperparas,
            use_optimizer=True,
            use_lr_scheduler=use_lr_scheduler,
            use_backbone=True,
            use_label_model=False,
            **kwargs
        )
        self.is_bert = self.config.backbone_config['name'] == 'BERT'
        if self.is_bert:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_config['paras']['model_name'])

    def fit(self,
            dataset_train: BaseDataset,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,
            cut_tied: Optional[bool] = False,
            evaluation_step: Optional[int] = 100,
            metric: Optional[Union[str, Callable]] = 'acc',
            direction: Optional[str] = 'auto',
            patience: Optional[int] = 20,
            tolerance: Optional[float] = -1.0,
            device: Optional[torch.device] = None,
            verbose: Optional[bool] = True,
            **kwargs: Any):

        if not verbose:
            logger.setLevel(logging.ERROR)

        config = self.config.update(**kwargs)
        hyperparas = self.config.hyperparas
        logger.info(config)

        n_steps = hyperparas['n_steps']
        if hyperparas['real_batch_size'] == -1 or hyperparas['batch_size'] < hyperparas['real_batch_size'] or not self.is_bert:
            hyperparas['real_batch_size'] = hyperparas['batch_size']
        accum_steps = hyperparas['batch_size'] // hyperparas['real_batch_size']

        n_rules = dataset_train.n_lf
        n_class = dataset_train.n_class
        balance = init_balance(n_class, dataset_valid=dataset_valid)

        backbone = self._init_model(
            dataset=dataset_train,
            n_class=dataset_train.n_class,
            config=config,
            is_bert=self.is_bert
        )
        model = WeaSELModel(
            input_size=dataset_train.features.shape[1],
            n_rules=n_rules,
            hidden_size=hyperparas['hidden_size'],
            n_class=n_class,
            temperature=hyperparas['temperature'],
            dropout=hyperparas['dropout'],
            backbone=backbone,
            balance=balance
        )
        self.model = model.to(device)

        optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model, config)

        labeled_dataset, _ = split_labeled_unlabeled(dataset_train, cut_tied=cut_tied)
        labeled_dataloader = self._init_train_dataloader(
            labeled_dataset,
            n_steps=n_steps,
            config=config,
            return_features=True,
            return_weak_labels=True,
        )

        valid_flag = self._init_valid_step(
            dataset_valid,
            y_valid,
            metric,
            direction,
            patience,
            tolerance,
            return_features=True,
            return_weak_labels=True,
        )

        if valid_flag:
            y = np.array(dataset_valid.labels)
            class_counts = Counter(y)
            sorted_counts = np.zeros(n_class)
            for c, cnt in class_counts.items():
                sorted_counts[c] = cnt
            balance = (sorted_counts + 1) / sum(sorted_counts)
            model.encoder.set_class_prior(balance)

        history = {}
        last_step_log = {}
        try:
            with trange(n_steps, desc="[TRAIN] WeaSEL", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for labeled_batch in labeled_dataloader:

                    loss = model.calculate_loss(labeled_batch)
                    loss.backward()
                    cnt += 1

                    if cnt % accum_steps == 0:
                        if hyperparas['grad_norm'] > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), hyperparas['grad_norm'])
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()
                        step += 1

                        if valid_flag and step % evaluation_step == 0:
                            metric_value, early_stop_flag, info = self._valid_step(step)
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history[step] = {
                                'loss'              : loss.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        if step >= n_steps:
                            break

        except KeyboardInterrupt:
            logger.info(f'KeyboardInterrupt! do not terminate the process in case need to save the best model')

        self._finalize()

        return history
