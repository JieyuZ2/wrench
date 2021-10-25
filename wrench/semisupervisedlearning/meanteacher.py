import logging
from typing import List, Any, Optional, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange

from .utils import exp_rampup, consistency_loss, BatchNormController
from ..backbone import BackBone
from ..basemodel import BaseTorchClassModel
from ..config import Config
from ..dataset import sample_batch, BaseDataset
from ..utils import cross_entropy_with_probs

logger = logging.getLogger(__name__)


class EMA:
    """
    Implementation from https://fyubang.com/2019/06/01/ema/
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class MeanTeacher(BaseTorchClassModel):
    def __init__(self,
                 ema_m: Optional[float] = 0.999,
                 lamb: Optional[float] = 0.4,
                 rampup_epochs: Optional[int] = 50,

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
            'ema_m'           : ema_m,  # momentum of exponential moving average
            'lamb'            : lamb,  # weight of unsupervised loss to supervised loss
            'rampup_epochs'   : rampup_epochs,  # rampup epochs for weight of unsupervised loss

            'batch_size'      : batch_size,
            'real_batch_size' : real_batch_size,
            'test_batch_size' : test_batch_size,
            'n_steps'         : n_steps,
            'grad_norm'       : grad_norm,
            'use_lr_scheduler': use_lr_scheduler,
            'binary_mode'     : False,
        }
        self.model: Optional[BackBone] = None
        self.ema: Optional[EMA] = None
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
            y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,
            include_labeled_as_unlabeled: Optional[bool] = False,
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
        n_steps_per_epoch = len(dataset_train) // hyperparas['real_batch_size']

        lamb = hyperparas['lamb']
        rampup_epochs = hyperparas['rampup_epochs']
        if include_labeled_as_unlabeled:
            labeled_dataset = dataset_train.create_subset(labeled_data_idx)
            unlabeled_dataset = dataset_train
        else:
            labeled_dataset, unlabeled_dataset = dataset_train.create_split(labeled_data_idx)

        if y_train is None:
            y_train = dataset_train.labels
        y_train = torch.Tensor([y_train[i] for i in labeled_data_idx]).to(device)

        model = self._init_model(
            dataset=dataset_train,
            n_class=dataset_train.n_class,
            config=config,
            is_bert=self.is_bert
        )
        self.model = model.to(device)

        self.ema = EMA(self.model, hyperparas['ema_m'])
        self.ema.register()

        bn_controller = BatchNormController()

        unlabeled_train_dataloader = self._init_train_dataloader(
            unlabeled_dataset,
            n_steps=0,
            config=config,
            drop_last=True
        )
        unlabeled_train_dataloader = sample_batch(unlabeled_train_dataloader)

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
            with trange(n_steps, desc="[TRAIN] MeanTeacher", unit="steps", disable=not verbose, ncols=200, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for labeled_batch in labeled_train_dataloader:
                    outputs = model(labeled_batch)
                    batch_idx = labeled_batch['ids'].to(device)
                    target = y_train[batch_idx]
                    loss_sup = cross_entropy_with_probs(outputs, target)

                    unlabeled_batch1 = next(unlabeled_train_dataloader)
                    unlabeled_batch2 = next(unlabeled_train_dataloader)

                    bn_controller.freeze_bn(self.model)
                    outputs1 = self.model(unlabeled_batch1)
                    bn_controller.unfreeze_bn(self.model)

                    self.ema.apply_shadow()
                    with torch.no_grad():
                        bn_controller.freeze_bn(self.model)
                        outputs2 = self.model(unlabeled_batch2)
                        bn_controller.unfreeze_bn(self.model)
                    self.ema.restore()

                    loss_unsup = consistency_loss(outputs1, outputs2)  # MSE loss for unlabeled data

                    loss = loss_sup + lamb * exp_rampup(cnt // n_steps_per_epoch, rampup_epochs) * loss_unsup
                    loss.backward()
                    cnt += 1

                    if cnt % accum_steps == 0:
                        # Clip the norm of the gradients.
                        if hyperparas['grad_norm'] > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), hyperparas['grad_norm'])
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        self.ema.update()
                        optimizer.zero_grad()
                        step += 1

                        if valid_flag and step % evaluation_step == 0:
                            metric_value, early_stop_flag, info = self._valid_step(step)
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history[step] = {
                                'loss'              : loss.item(),
                                'loss_sup'          : loss_sup.item(),
                                'loss_unsup'        : loss_unsup.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history[step])

                        last_step_log['loss'] = loss.item()
                        last_step_log['loss_sup'] = loss_sup.item()
                        last_step_log['loss_unsup'] = loss_unsup.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        if step >= n_steps:
                            break

        except KeyboardInterrupt:
            logger.info(f'KeyboardInterrupt! do not terminate the process in case need to save the best model')

        self._finalize()

        return history

    def predict_proba(self, *args: Any, **kwargs: Any):
        self.ema.apply_shadow()
        probas = super().predict_proba(*args, **kwargs)
        self.ema.restore()
        return probas
