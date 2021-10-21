import logging
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
from ..dataset import sample_batch, BaseDataset
from ..dataset.utils import split_labeled_unlabeled
from ..utils import cross_entropy_with_probs

logger = logging.getLogger(__name__)


def contrastive_loss(inputs, feat, margin=2.0, device=None):
    """copied from https://github.com/yueyu1030/COSINE/blob/main/trainer.py#L78"""
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(device)
    input_y = inputs[index, :]
    feat_y = feat[index, :]
    argmax_x = torch.argmax(inputs, dim=1)
    argmax_y = torch.argmax(input_y, dim=1)
    agreement = torch.FloatTensor([1 if x == True else 0 for x in argmax_x == argmax_y]).to(device)
    distances = (feat - feat_y).pow(2).mean(1)  # squared distances
    losses = 0.5 * (agreement * distances + (1 + -1 * agreement) * F.relu(margin - (distances + 1e-9).sqrt()).pow(2))
    return losses.mean()


def soft_frequency(logits, probs=False):
    """
    Unsupervised Deep Embedding for Clustering Analysis
    https://arxiv.org/abs/1511.06335
    """
    power = 2
    if not probs:
        softmax = nn.Softmax(dim=1)
        y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
    else:
        y = logits
    f = torch.sum(y, dim=0)
    t = y ** power / f
    t = t + 1e-10
    p = t / torch.sum(t, dim=-1, keepdim=True)
    return p


def calc_loss(inputs, target, reg=0.01):
    n_classes_ = inputs.shape[-1]
    loss_fn = nn.KLDivLoss(reduction='none')
    target = F.softmax(target, dim=1)
    weight = torch.sum(-torch.log(target + 1e-6) * target, dim=1)
    weight = 1 - weight / np.log(n_classes_)
    target = soft_frequency(target, probs=True)
    loss_batch = loss_fn(inputs, target)
    l = torch.sum(loss_batch * weight.unsqueeze(1))
    l -= reg * (torch.sum(inputs) + np.log(n_classes_) * n_classes_)
    return l


class Cosine(BaseTorchClassModel):
    def __init__(self,
                 teacher_update: Optional[int] = 100,
                 margin: Optional[float] = 1.0,
                 thresh: Optional[float] = 0.7,
                 mu: Optional[float] = 1.0,
                 lamda: Optional[float] = 0.1,

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
            'teacher_update'  : teacher_update,
            'margin'          : margin,
            'mu'              : mu,
            'thresh'          : thresh,
            'lamda'           : lamda,

            'batch_size'      : batch_size,
            'real_batch_size' : real_batch_size,
            'test_batch_size' : test_batch_size,
            'n_steps'         : n_steps,
            'grad_norm'       : grad_norm,
            'use_lr_scheduler': use_lr_scheduler,
            'binary_mode'     : binary_mode,
        }
        self.model: Optional[BackBone] = None
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

    def fit(self,
            dataset_train: BaseDataset,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,
            pretrained_model: str = None,
            cut_tied: Optional[bool] = False,
            soft_labels: Optional[bool] = False,
            evaluation_step: Optional[int] = 10,
            metric: Optional[Union[str, Callable]] = 'acc',
            direction: Optional[str] = 'auto',
            patience: Optional[int] = 10,
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

        teacher_update = hyperparas['teacher_update']
        margin = hyperparas['margin']
        thresh = hyperparas['thresh']
        mu = hyperparas['mu']
        lamda = hyperparas['lamda']

        assert config.backbone_config['name'] != 'LogReg'
        model = self._init_model(
            dataset=dataset_train,
            n_class=dataset_train.n_class,
            config=config,
            is_bert=self.is_bert
        )
        self.model = model.to(device)

        valid_flag = self._init_valid_step(dataset_valid, y_valid, metric, direction, patience, tolerance)
        history = {}

        if pretrained_model is not None:
            logger.info(f'loading pretrained model, so skip pretraining stage!')
            self.model.load_state_dict(pretrained_model)
        else:
            optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model, config)

            labeled_dataset, _ = split_labeled_unlabeled(dataset_train, cut_tied=cut_tied)
            labeled_dataloader = self._init_train_dataloader(
                labeled_dataset,
                n_steps=n_steps,
                config=config
            )

            label_model = self._init_label_model(config)
            label_model.fit(dataset_train=dataset_train, dataset_valid=dataset_valid, verbose=False)
            self.label_model = label_model
            if soft_labels:
                all_y_l = torch.FloatTensor(label_model.predict_proba(labeled_dataset)).to(device)
            else:
                all_y_l = torch.LongTensor(label_model.predict(labeled_dataset)).to(device)

            history_pretrain = {}
            last_step_log = {}
            with trange(n_steps, desc="[TRAIN] COSINE pretrain stage", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for batch in labeled_dataloader:
                    idx_l = batch['ids'].long().to(device)
                    y_l = all_y_l.index_select(0, idx_l)
                    predict_l = model(batch)
                    loss = cross_entropy_with_probs(predict_l, y_l)

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
                self.model.load_state_dict(self.best_model)

            history['pretrain'] = history_pretrain

        optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model, config)

        if valid_flag:
            self._reset_valid()
            self._valid_step(-1)
        history_selftrain = {}
        last_step_log = {}
        with trange(n_steps, desc="[TRAIN] COSINE distillation stage", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
            cnt = 0
            step = 0
            model.train()
            optimizer.zero_grad()
            while step < n_steps:

                if step % teacher_update == 0:
                    n = hyperparas['batch_size'] * teacher_update
                    sub_dataset, y_pseudo_l = self._get_new_dataset(
                        dataset_train,
                        n,
                        thresh
                    )
                    if sub_dataset is None:
                        logger.info(f'early stop because all the data are filtered!')
                        break

                    train_dataloader = self._init_train_dataloader(
                        sub_dataset,
                        n_steps=0,
                        config=config,
                    )

                    train_dataloader = sample_batch(train_dataloader)

                batch = next(train_dataloader)
                logits, f = model(batch, return_features=True)
                idx_l = batch['ids']
                y_pseudo = y_pseudo_l[idx_l].to(device)

                if logits.shape[1] == 1:
                    sigmoid_ = torch.sigmoid(logits)
                    log_softmax_logits = torch.cat([1 - sigmoid_, sigmoid_], -1)
                else:
                    log_softmax_logits = F.log_softmax(logits, dim=-1)

                loss_distill = calc_loss(inputs=log_softmax_logits,
                                         target=y_pseudo,
                                         reg=lamda,
                                         )

                loss_contrast = contrastive_loss(inputs=log_softmax_logits,
                                                 feat=f,
                                                 margin=margin,
                                                 device=device
                                                 )

                loss = loss_distill + mu * loss_contrast
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

                        history_selftrain[step] = {
                            'loss'              : loss.item(),
                            'loss_contrast'     : loss_contrast.item(),
                            'loss_distill'      : loss_distill.item(),
                            f'val_{metric}'     : metric_value,
                            f'best_val_{metric}': self.best_metric_value,
                            'best_step'         : self.best_step,
                        }
                        last_step_log.update(history_selftrain[step])

                    last_step_log['loss'] = loss.item()
                    last_step_log['loss_contrast'] = loss_contrast.item()
                    last_step_log['loss_distill'] = loss_distill.item()
                    pbar.update()
                    pbar.set_postfix(ordered_dict=last_step_log)

                    if step >= n_steps:
                        break

        self._finalize()

        history['selftrain'] = history_selftrain
        return history

    @torch.no_grad()
    def _get_new_dataset(self, dataset, n, thresh):
        self.model.eval()
        dataloader = self._init_valid_dataloader(dataset)
        model = self.model
        idx, y_pseudo = [], []
        constant = np.log(len(dataset.id2label))
        for batch in dataloader:
            output = model(batch)
            if output.shape[1] == 1:
                output = torch.sigmoid(output)
                proba = torch.cat([1 - output, output], -1)
            else:
                proba = F.softmax(output, dim=-1)
            weight = torch.sum(-torch.log(proba + 1e-5) * proba, dim=1)
            weight = 1 - weight / constant
            mask = weight > thresh

            idx += batch['ids'][mask].tolist()
            y_pseudo.append((proba[mask, :]).cpu())
            if len(idx) > n:
                break
        if len(idx) == 0:
            return None, None
        sub_dataset = dataset.create_subset(idx)
        y_pseudo = torch.cat(y_pseudo)
        self.model.train()
        return sub_dataset, y_pseudo
