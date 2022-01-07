import copy
import logging
from typing import Any, Optional, Union, Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import trange
from transformers import AutoTokenizer

from ..backbone import BackBone
from ..basemodel import BaseTorchClassModel
from ..config import Config
from ..dataset import BaseDataset
from ..utils import cross_entropy_with_probs

logger = logging.getLogger(__name__)

ABSTAIN = -1


class RuleAttentionTeacherNetwork(BackBone):
    def __init__(self, rule_embed_size, n_rules, n_class, hidden_size, dropout):
        super(RuleAttentionTeacherNetwork, self).__init__(n_class=n_class)
        self.rule_embedding = nn.Sequential(
            nn.Linear(rule_embed_size, n_rules),
            nn.Sigmoid(),
        )
        self.student_embedding = nn.Sequential(
            nn.Linear(rule_embed_size, 1),
            nn.Sigmoid(),
        )
        self.fcs = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, rule_embed_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, batch, features, proba):
        device = self.get_device()
        weak_labels = batch['weak_labels'].long().to(device)
        mask = weak_labels != ABSTAIN
        weak_labels_one_hot = F.one_hot(weak_labels * mask, num_classes=self.n_class)

        fc_h = self.fcs(features)
        rule_attention = self.rule_embedding(fc_h) * mask
        student_attention = self.student_embedding(fc_h)
        uniform_weight = torch.sum(mask, dim=1, keepdim=True) + 1 - student_attention - torch.sum(rule_attention, dim=1, keepdim=True)

        weighted_rule = rule_attention.unsqueeze(2) * weak_labels_one_hot
        weighted_proba = student_attention * torch.softmax(proba, dim=1)
        weighted_uniform = uniform_weight / self.n_class
        weighted_sum = torch.sum(weighted_rule, dim=1) + weighted_proba + weighted_uniform
        prediction = weighted_sum / torch.sum(weighted_sum, dim=1, keepdim=True)

        return prediction


class AstraModel(BackBone):
    def __init__(self, rule_embed_size, dropout, n_rules, n_class, backbone):
        super(AstraModel, self).__init__(n_class=n_class)
        self.backbone = backbone
        self.ran = RuleAttentionTeacherNetwork(rule_embed_size, n_rules, n_class, backbone.hidden_size, dropout)

    def forward_teacher(self, batch, features=None, proba=None):
        if features is None or proba is None:
            with torch.no_grad():
                proba, features = self.backbone(batch, return_features=True)
        return self.ran(batch, features, proba)

    def forward(self, batch, return_features=False):
        return self.backbone(batch, return_features)


def update_state_dict(model, state_dict: dict, mode: str):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(mode):
            new_state_dict[k[len(mode) + 1:]] = v
    getattr(model, mode).load_state_dict(new_state_dict)


class Astra(BaseTorchClassModel):
    def __init__(self,
                 n_iter: Optional[int] = 25,
                 outer_patience: Optional[int] = 3,
                 rule_embed_size: Optional[int] = 100,
                 dropout: Optional[float] = 0.3,

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
            'n_iter'          : n_iter,
            'outer_patience'  : outer_patience,
            'rule_embed_size' : rule_embed_size,
            'dropout'         : dropout,

            'batch_size'      : batch_size,
            'real_batch_size' : real_batch_size,
            'test_batch_size' : test_batch_size,
            'n_steps'         : n_steps,
            'grad_norm'       : grad_norm,
            'use_lr_scheduler': use_lr_scheduler,
            'binary_mode'     : binary_mode,
        }
        self.model: Optional[AstraModel] = None
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
            labeled_data_idx: List,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,
            pretrained_model: str = None,
            valid_mode: Optional[str] = 'student',
            soft_labels: Optional[bool] = False,
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

        n_steps = hyperparas['n_steps']
        if hyperparas['real_batch_size'] == -1 or hyperparas['batch_size'] < hyperparas['real_batch_size'] or not self.is_bert:
            hyperparas['real_batch_size'] = hyperparas['batch_size']
        accum_steps = hyperparas['batch_size'] // hyperparas['real_batch_size']

        assert config.backbone_config['name'] != 'LogReg'
        backbone = self._init_model(
            dataset=dataset_train,
            n_class=dataset_train.n_class,
            config=config,
            is_bert=self.is_bert
        )
        model = AstraModel(
            rule_embed_size=hyperparas['rule_embed_size'],
            dropout=hyperparas['dropout'],
            n_rules=n_rules,
            n_class=n_class,
            backbone=backbone,
        )
        self.model = model.to(device)

        labeled_dataloader = self._init_train_dataloader(
            labeled_dataset,
            n_steps=n_steps,
            config=config,
            return_weak_labels=True,
            return_labels=True,
        )

        unlabeled_dataloader = self._init_train_dataloader(
            unlabeled_dataset,
            n_steps=n_steps,
            config=config,
            return_weak_labels=True,
        )

        valid_flag = self._init_valid_step(
            dataset_valid,
            y_valid,
            metric,
            direction,
            patience,
            tolerance,
            return_weak_labels=True,
        )

        history = {}

        if pretrained_model is not None:
            logger.info(f'loading pretrained model, so skip pretraining stage!')
            self.model.backbone.load_state_dict(pretrained_model)
        else:
            optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model.backbone, config)

            history_pretrain = {}
            last_step_log = {}
            with trange(n_steps, desc="[TRAIN] ASTRA pretrain student", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for batch in labeled_dataloader:
                    predict_l = model(batch)
                    loss = cross_entropy_with_probs(predict_l, batch['labels'].to(device))
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
                            metric_value, early_stop_flag, info = self._valid_step(step, mode='student')
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history_pretrain[step] = {
                                'loss'              : loss.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history_pretrain[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        if step >= n_steps:
                            break

            if valid_flag:
                update_state_dict(self.model, self.best_model, 'backbone')

            history['pretrain'] = history_pretrain

        history_train = {}
        last_step_log = {}
        n_iter = hyperparas['n_iter']
        if valid_flag:
            outer_patience = hyperparas['outer_patience']
            outer_no_improve_cnt = 0
            outer_best_model = None
            if self.direction == 'maximize':
                outer_best_metric_value = -np.inf
            else:
                outer_best_metric_value = np.inf
        for i in range(n_iter):

            if valid_flag:
                self._reset_valid()
                self._valid_step(-1, mode='teacher')

            pseudo_probas_u, features_u = self.collect_pseudodataset_student(unlabeled_dataset)
            pseudo_probas_l, features_l = self.collect_pseudodataset_student(labeled_dataset)

            optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model.ran, config)

            history_train_teacher = {}
            with trange(n_steps, desc=f"[TRAIN@{i}] ASTRA-teacher", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for unlabeled_batch in unlabeled_dataloader:
                    idx_u = unlabeled_batch['ids'].long().to(device)
                    predict_u = model.forward_teacher(unlabeled_batch, features_u[idx_u], pseudo_probas_u[idx_u])
                    loss = - torch.mean(torch.sum(predict_u * torch.log(predict_u), dim=-1))
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
                            metric_value, early_stop_flag, info = self._valid_step(step, mode='teacher')
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history_train_teacher[step] = {
                                'loss'              : loss.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history_train_teacher[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        if step >= n_steps:
                            break

            if valid_flag:
                update_state_dict(self.model, self.best_model, 'ran')
                self._reset_valid()
                self._valid_step(-1, mode='teacher')

            optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model.ran, config)

            history_finetune_teacher = {}
            with trange(n_steps, desc=f"[FINETUNE@{i}] ASTRA-teacher", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for label_batch in labeled_dataloader:
                    idx_l = label_batch['ids'].long().to(device)
                    predict_l = model.forward_teacher(label_batch, features_l[idx_l], pseudo_probas_l[idx_l])
                    loss = cross_entropy_with_probs(predict_l, label_batch['labels'].to(device))
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
                            metric_value, early_stop_flag, info = self._valid_step(step, mode='teacher')
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history_finetune_teacher[step] = {
                                'loss'              : loss.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history_finetune_teacher[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        if step >= n_steps:
                            break

            if valid_flag:
                update_state_dict(self.model, self.best_model, 'ran')
                self._reset_valid()
                self._valid_step(-1, mode='student')

            optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model.backbone, config)

            pseudo_probas_u = self.collect_pseudodataset_teacher(unlabeled_dataset)
            if not soft_labels:
                pseudo_probas_u = torch.argmax(pseudo_probas_u, dim=-1)

            history_train_student = {}
            with trange(n_steps, desc=f"[TRAIN@{i}] ASTRA-student", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for unlabeled_batch in unlabeled_dataloader:
                    idx_u = unlabeled_batch['ids'].long().to(device)
                    predict_u = model(unlabeled_batch)
                    loss = cross_entropy_with_probs(predict_u, pseudo_probas_u[idx_u])
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
                            metric_value, early_stop_flag, info = self._valid_step(step, mode='student')
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history_train_student[step] = {
                                'loss'              : loss.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history_train_student[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        if step >= n_steps:
                            break

            if valid_flag:
                update_state_dict(self.model, self.best_model, 'backbone')
                self._reset_valid()
                self._valid_step(-1, mode='student')

            optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model.backbone, config)

            history_finetune_student = {}
            with trange(n_steps, desc=f"[FINETUNE@{i}] ASTRA-student", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for label_batch in labeled_dataloader:
                    predict_l = model(label_batch)
                    loss = cross_entropy_with_probs(predict_l, label_batch['labels'].to(device))
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
                            metric_value, early_stop_flag, info = self._valid_step(step, mode='student')
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history_finetune_student[step] = {
                                'loss'              : loss.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history_finetune_student[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        if step >= n_steps:
                            break

            if valid_flag:
                update_state_dict(self.model, self.best_model, 'backbone')
                metric_value, _, _ = self._valid_step(i, mode=valid_mode)
                if (self.direction == 'maximize' and metric_value > outer_best_metric_value) or \
                        (self.direction == 'minimize' and metric_value < outer_best_metric_value):
                    outer_best_metric_value = metric_value
                    outer_no_improve_cnt = 0
                    outer_best_model = copy.deepcopy(self.model.state_dict())
                else:
                    outer_no_improve_cnt += 1
                    if outer_patience > 0 and outer_no_improve_cnt >= outer_patience:
                        logger.info(f'[INFO] early stop outer loop @ iteration {i}')
                        break

            history_train[i] = {
                'train_teacher'   : history_train_teacher,
                'finetune_teacher': history_train_teacher,
                'train_student'   : history_train_teacher,
                'finetune_student': history_train_teacher,
            }

        self._finalize()
        if valid_flag:
            self.model.load_state_dict(outer_best_model)

        history['train'] = history_train
        return history

    @torch.no_grad()
    def collect_pseudodataset_student(self, dataset):
        model = self.model
        model.eval()
        if isinstance(dataset, BaseDataset):
            valid_dataloader = self._init_valid_dataloader(
                dataset,
            )
        else:
            valid_dataloader = dataset
        features, probas = [], []
        for batch in valid_dataloader:
            output, feature = model(batch, return_features=True)
            proba = F.softmax(output, dim=-1)
            probas.append(proba)
            features.append(feature)

        return torch.vstack(probas), torch.vstack(features)

    @torch.no_grad()
    def collect_pseudodataset_teacher(self, dataset):
        model = self.model
        model.eval()
        if isinstance(dataset, BaseDataset):
            valid_dataloader = self._init_valid_dataloader(
                dataset,
                return_weak_labels=True,
            )
        else:
            valid_dataloader = dataset
        probas = []
        for batch in valid_dataloader:
            proba = model.forward_teacher(batch)
            probas.append(proba)

        return torch.vstack(probas)

    @torch.no_grad()
    def predict_proba(self, dataset: Union[BaseDataset, DataLoader], mode: Optional[str] = 'student',
                      device: Optional[torch.device] = None, **kwargs: Any):
        assert mode in ['teacher', 'student'], f'mode: {mode} not support!'
        if device is not None:
            model = self.model.to(device)
        else:
            model = self.model
        model.eval()
        if isinstance(dataset, BaseDataset):
            valid_dataloader = self._init_valid_dataloader(
                dataset,
                return_weak_labels=mode == 'teacher',
            )
        else:
            valid_dataloader = dataset
        probas = []
        for batch in valid_dataloader:
            if mode == 'teacher':
                proba = model.forward_teacher(batch)
            elif mode == 'student':
                output = model(batch)
                proba = F.softmax(output, dim=-1)
            else:
                raise NotImplementedError

            probas.append(proba.cpu().numpy())

        return np.vstack(probas)
