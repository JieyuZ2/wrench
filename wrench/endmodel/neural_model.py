import logging
from typing import Any, Optional, Union, Callable

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import trange
from transformers import get_linear_schedule_with_warmup

from ..backbone import BackBone, MLP
from ..basemodel import BaseTorchClassModel
from ..dataset import BaseDataset, TorchDataset
from ..utils import cross_entropy_with_probs

logger = logging.getLogger(__name__)


class MLPModel(BaseTorchClassModel):
    def __init__(self,
                 lr: Optional[float] = 1e-3,
                 l2: Optional[float] = 1e-3,
                 hidden_size: Optional[int] = 100,
                 dropout: Optional[float] = 0.0,
                 batch_size: Optional[int] = 32,
                 test_batch_size: Optional[int] = 512,
                 n_steps: Optional[int] = 10000,
                 binary_mode: Optional[bool] = False
                 ):
        super().__init__()
        self.hyperparas = {
            'lr'             : lr,
            'l2'             : l2,
            'batch_size'     : batch_size,
            'test_batch_size': test_batch_size,
            'hidden_size'    : hidden_size,
            'dropout'        : dropout,
            'n_steps'        : n_steps,
            'binary_mode'    : binary_mode,
        }
        self.model: Optional[BackBone] = None

    def fit(self,
            dataset_train: BaseDataset,
            y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None,
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

        self._update_hyperparas(**kwargs)
        hyperparas = self.hyperparas

        n_steps = hyperparas['n_steps']
        train_dataloader = DataLoader(TorchDataset(dataset_train, n_data=n_steps * hyperparas['batch_size']),
                                      batch_size=hyperparas['batch_size'], shuffle=True)

        if y_train is None:
            y_train = dataset_train.labels
        y_train = torch.Tensor(y_train).to(device)

        if sample_weight is None:
            sample_weight = np.ones(len(dataset_train))
        sample_weight = torch.FloatTensor(sample_weight).to(device)

        n_class = dataset_train.n_class
        input_size = dataset_train.features.shape[1]
        model = MLP(
            input_size=input_size,
            n_class=n_class,
            hidden_size=hyperparas['hidden_size'],
            dropout=hyperparas['dropout'],
            binary_mode=hyperparas['binary_mode'],
        ).to(device)
        self.model = model

        optimizer = optim.Adam(model.parameters(), lr=hyperparas['lr'], weight_decay=hyperparas['l2'])

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=n_steps)

        valid_flag = self._init_valid_step(dataset_valid, y_valid, metric, direction, patience, tolerance)

        history = {}
        last_step_log = {}
        try:
            with trange(n_steps, desc="[TRAIN] MLP Classifier", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                model.train()
                step = 0
                for batch in train_dataloader:
                    step += 1
                    optimizer.zero_grad()
                    outputs = model(batch)
                    batch_idx = batch['ids'].to(device)
                    target = y_train[batch_idx]
                    loss = cross_entropy_with_probs(outputs, target, reduction='none')
                    loss = torch.mean(loss * sample_weight[batch_idx])
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

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
