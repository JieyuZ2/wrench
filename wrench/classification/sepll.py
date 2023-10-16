import logging
from typing import Any, Optional, Union, Callable, Dict

import copy

from tqdm.auto import trange
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from ..backbone import BackBone
from ..basemodel import BaseTorchClassModel, BaseLabelModel
from ..config import Config
from ..dataset import BaseDataset
from ..dataset.utils import split_labeled_unlabeled
from ..utils import cross_entropy_with_probs

logger = logging.getLogger(__name__)


def transform_snorkel_matrix_to_z_t(
        class_matrix: np.ndarray, col_row_dict: Dict = None
) -> [np.ndarray, np.ndarray, Dict]:
    """Takes a matrix in WRENCH format and transforms it to matrices Z and T which define rule matches and
    rule-to-class mappings.

    Input format:
        - class_matrix_ij = -1, iff the rule doesn't apply
        - class_matrix_ij = k, iff the rule labels class k

    :param class_matrix: shape=(num_samples, num_labelling_functions)
    :return: Z matrix - binary encoded array of which rules matched. Shape: instances x rules.
             T matrix - mapping of rules to classes, binary encoded. Shape: rules x classes.
             col_row_dict: Dictionary which connects WRENCH to SepLL format
    """

    if col_row_dict is None:
        col_row_dict = {}
        j = 0
        for i in range(class_matrix.shape[1]):
            col = set(class_matrix[:, i].tolist()) - {-1}
            if len(col) < 1:
                raise ValueError("Column has no matching classes")
            for elt in col:
                col_row_dict[j] = {
                    "original_col": i,
                    "label": elt
                }
                j += 1

    # init T matrix
    num_cols = len(col_row_dict)
    num_classes = int(max([elt.get("label") for elt in col_row_dict.values()]) + 1)
    t_matrix = np.zeros((num_cols, num_classes))

    for key, val in col_row_dict.items():
        t_matrix[key, val.get("label")] = 1

    # init Z matrix
    z_matrix = np.zeros((class_matrix.shape[0], num_cols))

    for j, val in col_row_dict.items():
        idx = np.where(class_matrix[:, val.get("original_col")] == val.get("label"))
        z_matrix[idx, j] = 1

    return z_matrix, t_matrix, col_row_dict


def merge_datasets(dataset_1: BaseDataset, dataset_2: BaseDataset):
    assert (dataset_1.n_lf == dataset_2.n_lf)
    assert (dataset_1.n_class == dataset_2.n_class)
    assert (len(dataset_1.id2label) == len(dataset_2.id2label))
    if isinstance(dataset_1.weak_labels, np.ndarray):
        dataset_1.weak_labels = dataset_2.weak_labels.tolist()

    for i in range(len(dataset_2.ids)):
        dataset_1.ids.append(dataset_2.ids[i])
        dataset_1.labels.append(dataset_2.labels[i])
        dataset_1.examples.append(dataset_2.examples[i])
        dataset_1.weak_labels.append(dataset_2.weak_labels[i])

    dataset_1.weak_labels = np.array(dataset_1.weak_labels)
    dataset_1.features = np.concatenate([dataset_1.features, dataset_2.features])
    return dataset_1


def transform_to_sepll_format(train_data: BaseDataset,
                              valid_data: Optional[BaseDataset] = None,
                              test_data: Optional[BaseDataset] = None):
    train_data_new = copy.deepcopy(train_data)
    train_matrix = np.array(train_data_new.weak_labels)

    train_z_matrix, t_matrix, col_row_dict = transform_snorkel_matrix_to_z_t(train_matrix)
    train_data_new.weak_labels = train_z_matrix
    train_data_new.t_matrix = t_matrix

    if valid_data is not None:
        valid_data_new = copy.deepcopy(valid_data)
        valid_z_matrix, _, _ = transform_snorkel_matrix_to_z_t(np.array(valid_data_new.weak_labels), col_row_dict)
        valid_data_new.weak_labels = valid_z_matrix
        valid_data_new.t_matrix = t_matrix
    else:
        valid_data_new = None

    if test_data is not None:
        test_data_new = copy.deepcopy(test_data)
        test_z_matrix, _, _ = transform_snorkel_matrix_to_z_t(np.array(test_data_new.weak_labels), col_row_dict)
        test_data_new.weak_labels = test_z_matrix
        test_data_new.t_matrix = t_matrix
    else:
        test_data_new = None

    return train_data_new, valid_data_new, test_data_new


class SepLLModel(nn.Module):
    def __init__(self, hidden_size: int, n_rules: int, n_class: int, T: torch.Tensor, dropout_proba: float, backbone):
        super(SepLLModel, self).__init__()

        self.n_class = n_class
        self.n_rules = n_rules

        self.backbone = backbone

        self.task_dropout = nn.Dropout(dropout_proba)
        self.task_classifier = nn.Linear(hidden_size, n_class, bias=False)

        self.lf_dropout = nn.Dropout(dropout_proba)
        self.lf_classifier = nn.Linear(hidden_size, n_rules)
        self.T = nn.Parameter(T, requires_grad=False)

    def forward(self, batch):
        _, pooler_output = self.backbone(batch, return_features=True)

        h = self.task_dropout(pooler_output)
        task_output = self.task_classifier(h)
        h_2 = self.lf_dropout(pooler_output)
        lf_output = self.lf_classifier(h_2)
        lf_pred_logspace = torch.matmul(task_output, self.T.T) + lf_output

        return torch.softmax(task_output, dim=-1), lf_output, lf_pred_logspace


class SepLL(BaseTorchClassModel):
    def __init__(self,
                 batch_size: Optional[int] = 16,
                 real_batch_size: Optional[int] = 16,
                 test_batch_size: Optional[int] = 16,
                 n_steps: Optional[int] = 10000,
                 grad_norm: Optional[float] = -1,
                 use_lr_scheduler: Optional[bool] = False,
                 binary_mode: Optional[bool] = False,
                 add_unlabeled: Optional[bool] = False,
                 class_noise: float = 0.0,
                 lf_l2_regularization: float = 0.0,
                 **kwargs: Any
                 ):
        super().__init__()
        self.hyperparas = {
            'batch_size': batch_size,
            'real_batch_size': real_batch_size,
            'test_batch_size': test_batch_size,
            'n_steps': n_steps,
            'grad_norm': grad_norm,
            'use_lr_scheduler': use_lr_scheduler,
            'binary_mode': binary_mode,
        }
        self.model: Optional[SepLLModel] = None
        self.label_model: Optional[BaseLabelModel] = None
        self.config = Config(
            self.hyperparas,
            use_optimizer=True,
            use_lr_scheduler=use_lr_scheduler,
            use_backbone=True,
            use_label_model=True,
            **kwargs
        )
        self.is_bert = self.config.backbone_config['name'] == 'BERT'
        if self.is_bert:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_config['paras']['model_name'])

        ## SepLL regularization settings:
        self.add_unlabeled = add_unlabeled
        self.class_noise = class_noise
        self.lf_l2_regularization = lf_l2_regularization

    def fit(self,
            dataset_train: BaseDataset,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,
            cut_tied: Optional[bool] = False,
            valid_mode: Optional[str] = 'feature',
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
        if hyperparas['real_batch_size'] == -1 or hyperparas['batch_size'] < hyperparas[
            'real_batch_size'] or not self.is_bert:
            hyperparas['real_batch_size'] = hyperparas['batch_size']
        accum_steps = hyperparas['batch_size'] // hyperparas['real_batch_size']

        # Data Preparaction
        dataset_train, dataset_valid, _ = transform_to_sepll_format(dataset_train, dataset_valid, test_data=None)

        labeled_dataset, unlabeled_dataset = split_labeled_unlabeled(dataset_train, cut_tied=cut_tied)
        if self.add_unlabeled:
            unlabeled_dataset.weak_labels = np.ones(unlabeled_dataset.weak_labels.shape)
            labeled_dataset = merge_datasets(labeled_dataset, unlabeled_dataset)
        if self.class_noise > 0:
            T = labeled_dataset.t_matrix
            Z = labeled_dataset.weak_labels

            exist_classes = np.dot(Z, T) > 0
            sample_classes = np.dot(exist_classes, T.T)

            sample = np.random.binomial(n=1, p=self.class_noise, size=Z.shape)
            Z_new = Z + sample * sample_classes
            np.clip(Z_new.round(), a_min=0, a_max=1)
            labeled_dataset.weak_labels = Z_new

        labeled_dataset.weak_labels = np.nan_to_num(
            labeled_dataset.weak_labels / labeled_dataset.weak_labels.sum(axis=1, keepdims=True)
        )

        dataset_valid.weak_labels = np.nan_to_num(
            dataset_valid.weak_labels / dataset_valid.weak_labels.sum(axis=1, keepdims=True)
        )

        # Model Preparation

        T = torch.from_numpy(dataset_train.t_matrix).float()
        n_rules = T.shape[0]
        n_class = T.shape[1]

        backbone = self._init_model(
            dataset=dataset_train,
            n_class=dataset_train.n_class,
            config=config,
            is_bert=self.is_bert
        )
        if self.is_bert:
            hidden_size = dataset_train.features.shape[1]
        else:
            hidden_size = self.config.backbone_config["paras"]["hidden_size"]

        model = SepLLModel(
            hidden_size=hidden_size,
            n_rules=n_rules,
            n_class=n_class,
            T=T,
            dropout_proba=0.1,
            backbone=backbone
        )
        self.model = model.to(device)

        # Train Loop Preparation

        optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model, config)

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

        history = {}
        last_step_log = {}
        try:
            with trange(n_steps, desc="[TRAIN] SepLL", unit="steps", disable=not verbose, ncols=150,
                        position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for batch in labeled_dataloader:

                    z = batch['weak_labels'].to(device)
                    task_output, lf_output, lf_pred_logspace = model(batch)
                    loss = cross_entropy_with_probs(lf_pred_logspace, z)
                    if self.lf_l2_regularization > 0:
                        lf_weight = model.lf_classifier.weight
                        loss += self.lf_l2_regularization * (lf_weight * lf_weight).sum()
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
                            metric_value, early_stop_flag, info = self._valid_step(step, mode=valid_mode)
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history[step] = {
                                'loss': loss.item(),
                                f'val_{metric}': metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step': self.best_step,
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
    def predict_proba(self, dataset: Union[BaseDataset, DataLoader], device: Optional[torch.device] = None,
                      **kwargs: Any):
        if device is not None:
            model = self.model.to(device)
        else:
            model = self.model
        model.eval()
        if isinstance(dataset, BaseDataset):
            valid_dataloader = self._init_valid_dataloader(
                dataset,
                return_features=True,
                return_weak_labels=True,
            )
        else:
            valid_dataloader = dataset
        probas = []
        for batch in valid_dataloader:
            predict_proba, _, _ = model(batch)
            probas.append(predict_proba.cpu().numpy())

        return np.vstack(probas)
