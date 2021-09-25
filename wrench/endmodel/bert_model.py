import logging
from typing import Any, Optional, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import trange
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer

from ..backbone import BackBone
from ..basemodel import BaseTorchClassModel
from ..dataset import BaseDataset
from ..utils import cross_entropy_with_probs, get_bert_model_class, get_bert_torch_dataset_class, construct_collate_fn_trunc_pad

logger = logging.getLogger(__name__)

collate_fn = construct_collate_fn_trunc_pad('mask')


class BertClassifierModel(BaseTorchClassModel):
    def __init__(self,
                 model_name: Optional[str] = 'bert-base-cased',
                 lr: Optional[float] = 3e-5,
                 l2: Optional[float] = 0.0,
                 max_tokens: Optional[int] = 512,
                 batch_size: Optional[int] = 16,
                 real_batch_size: Optional[int] = 16,
                 test_batch_size: Optional[int] = 16,
                 n_steps: Optional[int] = 10000,
                 fine_tune_layers: Optional[int] = -1,
                 binary_mode: Optional[bool] = False,
                 ):
        super().__init__()
        self.hyperparas = {
            'model_name'      : model_name,
            'fine_tune_layers': fine_tune_layers,
            'lr'              : lr,
            'l2'              : l2,
            'max_tokens'      : max_tokens,
            'batch_size'      : batch_size,
            'real_batch_size' : real_batch_size,
            'test_batch_size' : test_batch_size,
            'n_steps'         : n_steps,
            'binary_mode'     : binary_mode,
        }
        self.model: Optional[BackBone] = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def fit(self,
            dataset_train: BaseDataset,
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
        if hyperparas['real_batch_size'] == -1 or hyperparas['batch_size'] < hyperparas['real_batch_size']:
            hyperparas['real_batch_size'] = hyperparas['batch_size']
        accum_steps = hyperparas['batch_size'] // hyperparas['real_batch_size']
        torch_dataset = get_bert_torch_dataset_class(dataset_train)(dataset_train, self.tokenizer, self.hyperparas['max_tokens'],
                                                                    n_data=n_steps * hyperparas['batch_size'])
        train_dataloader = DataLoader(torch_dataset, batch_size=hyperparas['real_batch_size'], shuffle=True, collate_fn=collate_fn)

        if y_train is None:
            y_train = dataset_train.labels
        y_train = torch.Tensor(y_train).to(device)

        if sample_weight is None:
            sample_weight = np.ones(len(dataset_train))
        sample_weight = torch.FloatTensor(sample_weight).to(device)

        n_class = dataset_train.n_class
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
            with trange(n_steps, desc=f"[FINETUNE] {hyperparas['model_name']} Classifier", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for batch in train_dataloader:
                    outputs = model(batch)
                    batch_idx = batch['ids'].to(device)
                    target = y_train[batch_idx]
                    loss = cross_entropy_with_probs(outputs, target, reduction='none')
                    loss = torch.mean(loss * sample_weight[batch_idx])
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
