import logging
from typing import Any, Optional, Union, Callable

import higher
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import trange
from transformers import AutoTokenizer

from ..backbone import BackBone
from ..basemodel import BaseTorchClassModel
from ..config import Config
from ..dataset import sample_batch, BaseDataset
from ..utils import cross_entropy_with_probs

logger = logging.getLogger(__name__)


class LearningToReweight(BaseTorchClassModel):
    def __init__(self,
                 batch_size: Optional[int] = 16,
                 test_batch_size: Optional[int] = 16,
                 n_steps: Optional[int] = 10000,
                 grad_norm: Optional[float] = -1,
                 use_lr_scheduler: Optional[bool] = False,
                 binary_mode: Optional[bool] = False,
                 **kwargs: Any
                 ):
        super().__init__()
        self.hyperparas = {
            'batch_size'      : batch_size,
            'real_batch_size' : batch_size,
            'test_batch_size' : test_batch_size,
            'n_steps'         : n_steps,
            'grad_norm'       : grad_norm,
            'use_lr_scheduler': use_lr_scheduler,
            'binary_mode'     : binary_mode,
        }
        self.model: Optional[BackBone] = None
        self.config = Config(
            self.hyperparas,
            use_optimizer=True,
            use_lr_scheduler=use_lr_scheduler,
            use_backbone=True,
            **kwargs
        )
        self.is_bert = self.config.backbone_config['name'] == 'BERT'
        if self.is_bert:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_config['paras']['model_name'])

    def fit(self,
            dataset_train: BaseDataset,
            y_train: Optional[np.ndarray] = None,
            dataset_valid: BaseDataset = None,
            y_valid: Optional[np.ndarray] = None,
            evaluation_step: Optional[int] = 10,
            metric: Optional[Union[str, Callable]] = 'acc',
            direction: Optional[str] = 'auto',
            patience: Optional[int] = 100,
            tolerance: Optional[float] = -1.0,
            device: Optional[torch.device] = None,
            verbose: Optional[bool] = True,
            **kwargs: Any):

        assert dataset_valid is not None

        if not verbose:
            logger.setLevel(logging.ERROR)

        config = self.config.update(**kwargs)
        hyperparas = self.config.hyperparas
        logger.info(config)

        n_steps = hyperparas['n_steps']

        if y_train is None:
            y_train = dataset_train.labels
        y_train = torch.Tensor(y_train).to(device)

        model = self._init_model(
            dataset=dataset_train,
            n_class=dataset_train.n_class,
            config=config,
            is_bert=self.is_bert
        )
        self.model = model.to(device)

        train_dataloader = self._init_train_dataloader(
            dataset_train,
            n_steps=n_steps,
            config=config
        )

        optimizer, scheduler = self._init_optimizer_and_lr_scheduler(model, config)

        train_meta_dataloader = self._init_train_dataloader(
            dataset_valid,
            n_steps=n_steps,
            config=config
        )
        train_meta_dataloader = sample_batch(train_meta_dataloader)

        valid_flag = self._init_valid_step(dataset_valid, y_valid, metric, direction, patience, tolerance)

        history = {}
        last_step_log = {}
        try:
            with trange(n_steps, desc=f"[TRAIN]", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                step = 0
                model.train()
                for batch in train_dataloader:
                    step += 1

                    batch_idx = batch['ids'].to(device)
                    target = y_train[batch_idx]

                    weak_logits = model(batch)

                    with higher.innerloop_ctx(model, optimizer, device=device) as (meta_model, diffopt):

                        # updates meta model on weak data
                        meta_logits = meta_model(batch)
                        meta_train_loss = cross_entropy_with_probs(meta_logits, target, reduction="none")
                        eps = torch.zeros(len(target), requires_grad=True, device=device)
                        meta_train_loss = torch.sum(eps * meta_train_loss)

                        diffopt.step(meta_train_loss)

                        # compute grads of eps on strong data
                        meta_batch = next(train_meta_dataloader)
                        meta_loss = cross_entropy_with_probs(meta_model(meta_batch), meta_batch['labels'].to(device))

                        eps_grads = torch.autograd.grad(meta_loss, eps)[0].detach()

                    w_tilde = F.normalize(torch.clamp(-eps_grads, min=0), p=1, dim=0)

                    # performs update on weighted batch of weak data
                    loss = torch.sum(w_tilde * cross_entropy_with_probs(weak_logits, target, reduction="none"))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    if valid_flag and step % evaluation_step == 0:
                        metric_value, early_stop_flag, info = self._valid_step(step)
                        if early_stop_flag:
                            logger.info(info)
                            break

                        history[step] = {
                            'loss'              : loss.item(),
                            'meta_loss'         : meta_loss.item(),
                            f'val_{metric}'     : metric_value,
                            f'best_val_{metric}': self.best_metric_value,
                            'best_step'         : self.best_step,
                        }
                        last_step_log.update(history[step])

                    last_step_log['loss'] = loss.item()
                    last_step_log['meta_loss'] = meta_loss.item()
                    pbar.update()
                    pbar.set_postfix(ordered_dict=last_step_log)

                    if step >= n_steps:
                        break

        except KeyboardInterrupt:
            logger.info(f'KeyboardInterrupt! do not terminate the process in case need to save the best model')

        self._finalize()

        return history
