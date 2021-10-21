import logging
from typing import Any, Optional, Union, Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import trange

from ..backbone import BackBone
from ..basemodel import BaseTorchClassModel
from ..config import Config
from ..dataset import BaseDataset
from ..dataset.utils import split_labeled_unlabeled
from ..utils import cross_entropy_with_probs

logger = logging.getLogger(__name__)

ABSTAIN = -1


class RuleNetwork(nn.Module):
    def __init__(self, input_size, n_rules, hidden_size, dropout=0.8):
        super(RuleNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size + n_rules, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        z = self.model(x)
        score = torch.sigmoid(z)
        return score


class ImplyLossModel(BackBone):
    def __init__(self, input_size, n_rules, n_class, backbone, hidden_size, q, dropout=0.8):
        super(ImplyLossModel, self).__init__(n_class=n_class)
        self.backbone = backbone
        self.rule_network = RuleNetwork(input_size, n_rules, hidden_size, dropout)
        self.n_rules = n_rules
        self.rule_embedding = nn.Parameter(torch.eye(n_rules), requires_grad=False)
        self.q = q

    def get_r_score(self, x):
        covered_features = torch.repeat_interleave(x, self.n_rules, dim=0)
        rule_embedding = self.rule_embedding.repeat(x.size(0), 1)
        concat_feature = torch.cat((covered_features, rule_embedding), dim=1)
        r_score = self.rule_network(concat_feature).view(-1, self.n_rules)
        return r_score

    def forward(self, batch):
        device = self.get_device()
        proba = torch.softmax(self.backbone(batch), dim=1)
        weak_labels_list = batch['weak_labels'].to(device).long()
        features = batch['features'].to(device)

        covered_mask = torch.sum(weak_labels_list != ABSTAIN, dim=1) > 0
        if torch.any(covered_mask):
            covered_features = features[covered_mask]
            covered_weak_labels_list = weak_labels_list[covered_mask]
            batch_size = covered_features.size(0)

            covered_features = torch.repeat_interleave(covered_features, self.n_rules, dim=0)
            rule_embedding = self.rule_embedding.repeat(batch_size, 1)
            concat_feature = torch.cat((covered_features, rule_embedding), dim=1)
            r_score = self.rule_network(concat_feature).view(-1, self.n_rules)

            fire_mask = (covered_weak_labels_list != ABSTAIN) & (r_score > 0.5)
            weak_labels_one_hot = F.one_hot(covered_weak_labels_list * fire_mask, num_classes=self.n_class)
            score = weak_labels_one_hot * r_score.unsqueeze(2) + (1 - weak_labels_one_hot) * (1 - r_score.unsqueeze(2))
            score_mean = torch.sum(score * fire_mask.unsqueeze(2), dim=1) / torch.sum(fire_mask, dim=1, keepdim=True)

            proba[covered_mask] += torch.nan_to_num(score_mean, nan=0)

        return proba

    def calculate_labeled_batch_loss(self, labeled_batch, data_exemplar_matrix):
        device = self.get_device()
        y_l = labeled_batch['labels'].to(device)
        feature_l = labeled_batch['features'].to(device)
        idx_l = labeled_batch['ids']
        weak_labels = labeled_batch['weak_labels'].long().to(device)

        exemplar_mask = data_exemplar_matrix[idx_l]
        fire_mask = (weak_labels != ABSTAIN) & (~exemplar_mask)
        equal_mask = weak_labels == y_l.unsqueeze(1)

        r_score = self.get_r_score(feature_l)

        # Eq (2) first term
        r_score_l = r_score.masked_select(exemplar_mask)
        loss_phi_1 = F.binary_cross_entropy(r_score_l, torch.ones_like(r_score_l), reduction='mean')

        # Eq (2) second term
        mask = fire_mask & (~equal_mask)
        if torch.sum(mask):
            r_score_l = r_score.masked_select(mask)
            loss_phi_2 = F.binary_cross_entropy(r_score_l, torch.zeros_like(r_score_l), reduction='mean')
        else:
            loss_phi_2 = 0.0

        # Eq (2) third term
        mask = fire_mask & equal_mask
        if torch.sum(mask):
            r_score_l = r_score.masked_select(mask)
            loss_phi_3 = torch.mean(1 - torch.pow(r_score_l, self.q)) / self.q
        else:
            loss_phi_3 = 0.0

        # Eq (1)
        predict_l = self.backbone(labeled_batch)
        loss_theta = cross_entropy_with_probs(predict_l, y_l, reduction='mean')

        loss = loss_theta + loss_phi_1 + loss_phi_2 + loss_phi_3

        return loss

    def calculate_unlabeled_batch_loss(self, unlabeled_batch):
        device = self.get_device()
        feature_u = unlabeled_batch['features'].to(device)
        weak_labels = unlabeled_batch['weak_labels'].long().to(device)
        batch_size = feature_u.size(0)

        # Eq (4)
        r_score = self.get_r_score(feature_u)
        proba = torch.softmax(self.backbone(unlabeled_batch), dim=1)
        proba_expand = proba[torch.arange(batch_size).unsqueeze(1), weak_labels]

        mask = weak_labels != ABSTAIN
        score = 1 - r_score.masked_select(mask) * (1 - proba_expand.masked_select(mask))
        loss = F.binary_cross_entropy(score, torch.ones_like(score))

        return loss


class ImplyLoss(BaseTorchClassModel):
    def __init__(self,
                 hidden_size: Optional[int] = 100,
                 dropout: Optional[float] = 0.8,
                 q: Optional[float] = 0.2,
                 gamma: Optional[float] = 0.1,

                 batch_size: Optional[int] = 32,
                 real_batch_size: Optional[int] = 32,
                 test_batch_size: Optional[int] = 16,
                 n_steps: Optional[int] = 10000,
                 grad_norm: Optional[float] = -1,
                 use_lr_scheduler: Optional[bool] = False,
                 **kwargs: Any
                 ):
        super().__init__()
        self.hyperparas = {
            'hidden_size'     : hidden_size,
            'dropout'         : dropout,
            'q'               : q,  # q for Generalized-XENT loss
            'gamma'           : gamma,  # weighting factor for loss on U

            'batch_size'      : batch_size,
            'real_batch_size' : real_batch_size,
            'test_batch_size' : test_batch_size,
            'n_steps'         : n_steps,
            'grad_norm'       : grad_norm,
            'use_lr_scheduler': use_lr_scheduler,
            'binary_mode'     : False,
        }
        self.model: Optional[ImplyLossModel] = None
        self.config = Config(
            self.hyperparas,
            use_optimizer=True,
            use_lr_scheduler=use_lr_scheduler,
            use_backbone=True,
            use_label_model=False,
            **kwargs
        )
        self.is_bert = False

    def fit(self,
            dataset_train: BaseDataset,
            labeled_data_idx: List,
            exemplar_idx: List,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,
            valid_mode: Optional[str] = 'implyloss',
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

        n_rules = dataset_train.n_lf
        n_class = dataset_train.n_class
        labeled_dataset, unlabeled_dataset = dataset_train.create_split(labeled_data_idx)
        unlabeled_dataset, _ = split_labeled_unlabeled(unlabeled_dataset)

        data_exemplar_matrix = np.zeros((len(labeled_data_idx), n_rules))
        for rule_id, data_id in enumerate(exemplar_idx):
            new_data_id = labeled_data_idx.index(data_id)
            data_exemplar_matrix[new_data_id, rule_id] = 1
        data_exemplar_matrix = torch.BoolTensor(data_exemplar_matrix).to(device)

        n_steps = hyperparas['n_steps']
        if hyperparas['real_batch_size'] == -1 or hyperparas['batch_size'] < hyperparas['real_batch_size'] or not self.is_bert:
            hyperparas['real_batch_size'] = hyperparas['batch_size']
        accum_steps = hyperparas['batch_size'] // hyperparas['real_batch_size']

        backbone = self._init_model(
            dataset=dataset_train,
            n_class=dataset_train.n_class,
            config=config,
            is_bert=self.is_bert
        )
        model = ImplyLossModel(
            input_size=dataset_train.features.shape[1],
            n_rules=n_rules,
            n_class=n_class,
            backbone=backbone,
            hidden_size=hyperparas['hidden_size'],
            q=hyperparas['q'],
            dropout=hyperparas['dropout'],
        )
        self.model = model.to(device)

        unlabeled_train_dataloader = self._init_train_dataloader(
            unlabeled_dataset,
            n_steps=n_steps,
            config=config
        )

        labeled_train_dataloader = self._init_train_dataloader(
            labeled_dataset,
            n_steps=n_steps,
            config=config
        )

        optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model, config)

        valid_flag = self._init_valid_step(dataset_valid, y_valid, metric, direction, patience, tolerance)

        history = {}
        last_step_log = {}
        try:
            with trange(n_steps, desc="[TRAIN] ImplyLoss", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for unlabeled_batch, labeled_batch in zip(unlabeled_train_dataloader, labeled_train_dataloader):

                    loss_u = model.calculate_unlabeled_batch_loss(
                        unlabeled_batch,
                    )

                    loss_l = model.calculate_labeled_batch_loss(
                        labeled_batch,
                        data_exemplar_matrix,
                    )

                    loss = loss_l + hyperparas['gamma'] * loss_u
                    loss.backward()
                    cnt += 1

                    if cnt % accum_steps == 0:
                        # Clip the norm of the gradients.
                        if hyperparas['grad_norm'] > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), hyperparas['grad_norm'])
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()
                        step += 1

                        if valid_flag and step % evaluation_step == 0:
                            metric_value, early_stop_flag, info = self._valid_step(step, mode=valid_mode)
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

    @torch.no_grad()
    def predict_proba(self, dataset: Union[BaseDataset, DataLoader], mode: Optional[str] = 'implyloss',
                      device: Optional[torch.device] = None, **kwargs: Any):
        assert mode in ['classifier', 'implyloss'], f'mode: {mode} not support!'
        if device is not None:
            model = self.model.to(device)
        else:
            model = self.model
        model.eval()
        if isinstance(dataset, BaseDataset):
            valid_dataloader = self._init_valid_dataloader(dataset)
        else:
            valid_dataloader = dataset
        probas = []
        for batch in valid_dataloader:
            if mode == 'classifier':
                logits = model.backbone(batch)
                proba = torch.softmax(logits, dim=-1)
            elif mode == 'implyloss':
                proba = torch.softmax(model(batch), dim=-1)
            else:
                raise NotImplementedError

            probas.append(proba.cpu().numpy())

        return np.vstack(probas)
