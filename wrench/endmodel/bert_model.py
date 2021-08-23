from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
from tqdm import tqdm, trange
import numpy as np

from ..utils import cross_entropy_with_probs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AdamW, get_linear_schedule_with_warmup

from ..basemodel import BaseModel, BaseTorchModel
from ..dataset import BaseDataset, TorchDataset
from ..utils import get_bert_model_class


logger = logging.getLogger(__name__)


class BertClassifierModel(BaseTorchModel):
    def __init__(self,
                 model_name: Optional[str] = 'bert-base-cased',
                 lr: Optional[float] = 3e-5,
                 l2: Optional[float] = 0.0,
                 batch_size: Optional[int] = 16,
                 real_batch_size: Optional[int] = 16,
                 test_batch_size: Optional[int] = 16,
                 n_steps: Optional[int] = 10000,
                 fine_tune_layers: Optional[int] = -1,
                 binary_mode: Optional[bool] = False,
                 ):
        super().__init__()
        self.hyperparas = {
            'model_name': model_name,
            'fine_tune_layers': fine_tune_layers,
            'lr': lr,
            'l2': l2,
            'batch_size': batch_size,
            'real_batch_size': real_batch_size,
            'test_batch_size': test_batch_size,
            'n_steps': n_steps,
            'binary_mode': binary_mode,
        }
        self.model: Optional[BaseModel] = None

    def fit(self,
            dataset_train:BaseDataset,
            y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None,
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

        self._update_hyperparas(**kwargs)
        hyperparas = self.hyperparas

        n_steps = hyperparas['n_steps']
        accum_steps = hyperparas['batch_size'] // hyperparas['real_batch_size']
        train_dataloader = DataLoader(TorchDataset(dataset_train, n_data=n_steps*hyperparas['batch_size']),
                                      batch_size=hyperparas['real_batch_size'], shuffle=True)

        if y_train is not None:
            y_train = torch.Tensor(y_train).to(device)

        if sample_weight is None:
            sample_weight = np.ones(len(dataset_train))
        sample_weight = torch.FloatTensor(sample_weight).to(device)

        n_class = len(dataset_train.id2label)
        model = get_bert_model_class(dataset_train)(
            n_class=n_class,
            **hyperparas
        ).to(device)
        self.model = model

        optimizer = AdamW(model.parameters(), lr=hyperparas['lr'], weight_decay=hyperparas['l2'])

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=n_steps)

        valid_flag = self._init_valid_step(dataset_valid, y_valid, metric, direction, patience, tolerance)

        history = {}
        last_step_log = {}
        try:
            with tqdm(total=n_steps, desc=f"Finetuing {hyperparas['model_name']} model:",
                        unit="steps", disable=not verbose, ncols=200) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for batch in train_dataloader:
                    outputs = model(batch)
                    batch_idx = batch['ids'].to(device)
                    if y_train is not None:
                        target = y_train[batch_idx]
                    else:
                        target = batch['labels'].to(device)
                    loss = cross_entropy_with_probs(outputs, target, reduction='none')
                    batch_sample_weights = sample_weight[batch_idx]
                    loss = torch.mean(loss * batch_sample_weights)
                    loss.backward()
                    cnt += 1

                    if cnt % accum_steps == 0:
                        # Clip the norm of the gradients to 1.0.
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        step += 1

                        if valid_flag and step % evaluation_step == 0:
                            metric_value, early_stop_flag, info = self._valid_step(step)
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history[step] = {
                                'loss': loss.item(),
                                f'val_{metric}': metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                f'best_step': self.best_step,
                            }
                            last_step_log.update(history[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)
        except KeyboardInterrupt:
            logger.info(f'KeyboardInterrupt! do not terminate the process in case need to save the best model')

        self._finalize()

        return history