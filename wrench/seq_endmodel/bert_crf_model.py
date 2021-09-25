import logging
import math
from copy import deepcopy
from typing import Any, List, Optional, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer

from ..backbone import BertSeqTagger
from ..basemodel import BaseTorchSeqModel
from ..dataset.seqdataset import BaseSeqDataset
from ..utils import construct_collate_fn_trunc_pad

logger = logging.getLogger(__name__)

collate_fn = construct_collate_fn_trunc_pad('mask')


class BERTTorchSeqDataset(Dataset):
    def __init__(self, dataset: BaseSeqDataset, tokenizer, max_seq_length, use_crf, n_data: Optional[int] = 0):
        self.id2label = deepcopy(dataset.id2label)
        self.label2id = deepcopy(dataset.label2id)
        self.n_class = len(self.id2label)

        if not use_crf:
            self.dum_label = 'X'
            self.label2id[self.dum_label] = len(self.id2label)
            self.id2label.append(self.dum_label)

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length  # set to -1 when test
        self.use_crf = use_crf

        corpus = list(map(lambda x: x["text"], dataset.examples))
        self.seq_len = list(map(len, corpus))
        input_ids_tensor, input_mask_tensor, predict_mask_tensor = self.convert_corpus_to_tensor(corpus)

        self.input_ids_tensor = input_ids_tensor
        self.input_mask_tensor = input_mask_tensor
        self.predict_mask_tensor = predict_mask_tensor

        n_data_ = len(input_ids_tensor)
        self.n_data_ = n_data_
        if n_data > 0:
            self.n_data = math.ceil(n_data / n_data_) * n_data_
        else:
            self.n_data = n_data_

    def __len__(self):
        return self.n_data

    def convert_corpus_to_tensor(self, corpus):
        input_ids_list = []
        input_mask_list = []
        predict_mask_list = []
        max_seq_length = 0

        for words in corpus:
            predict_mask = []
            input_mask = []
            tokens = []

            for i, w in enumerate(words):
                sub_words = self.tokenizer.tokenize(w)
                if not sub_words:
                    sub_words = [self.tokenizer.unk_token]
                if self.use_crf:
                    ''' if crf is used, then the padded token will be ignored '''
                    tokens.append(sub_words[0])
                else:
                    tokens.extend(sub_words)
                for j in range(len(sub_words)):
                    if j == 0:
                        input_mask.append(1)
                        predict_mask.append(1)
                    elif not self.use_crf:  # These padding will hurt performance
                        ''' '##xxx' -> 'X' (see bert paper, for non-crf model only) '''
                        input_mask.append(1)
                        predict_mask.append(0)

            max_seq_length = max(max_seq_length, len(tokens))
            input_ids_list.append(self.tokenizer.convert_tokens_to_ids(tokens))
            input_mask_list.append(input_mask)
            predict_mask_list.append(predict_mask)

        max_seq_length = min(max_seq_length, self.max_seq_length)

        n = len(input_ids_list)
        for i in range(n):
            ni = len(input_ids_list[i])
            if ni > max_seq_length:
                logger.info(f'Example is too long, length is {ni}, truncated to {max_seq_length}!')
                input_ids_list[i] = input_ids_list[i][:max_seq_length]
                input_mask_list[i] = input_mask_list[i][:max_seq_length]
                predict_mask_list[i] = predict_mask_list[i][:max_seq_length]
            else:
                input_ids_list[i].extend([self.tokenizer.pad_token_id] * (max_seq_length - ni))
                input_mask_list[i].extend([0] * (max_seq_length - ni))
                predict_mask_list[i].extend([0] * (max_seq_length - ni))

        input_ids_tensor = torch.LongTensor(input_ids_list)
        input_mask_tensor = torch.LongTensor(input_mask_list)
        predict_mask_tensor = torch.LongTensor(predict_mask_list)

        return input_ids_tensor, input_mask_tensor, predict_mask_tensor

    def prepare_labels(self, labels):
        O_id = self.label2id['O']
        if self.use_crf:
            n, max_seq_len = self.predict_mask_tensor.shape
            prepared_labels = np.ones((n, max_seq_len), dtype=int) * O_id
            for i, labels_i in enumerate(labels):
                ni = len(labels_i)
                if ni > max_seq_len:
                    prepared_labels[i, :] = labels_i[:max_seq_len]
                else:
                    prepared_labels[i, :ni] = labels_i
        else:
            prepared_labels = []
            add_label_id = self.label2id[self.dum_label]
            for labels_i, mask in zip(labels, self.predict_mask_tensor):
                pre_labels = []
                cnt = 0
                n = len(labels_i)
                for idx, flag in enumerate(mask):
                    if flag:
                        pre_labels.append(labels_i[cnt])
                        cnt += 1
                    else:
                        if n == cnt:
                            pre_labels.append(O_id)
                        else:
                            pre_labels.append(add_label_id)
                prepared_labels.append(pre_labels)

        return torch.LongTensor(prepared_labels)

    def __getitem__(self, idx):
        idx = idx % self.n_data_
        d = {
            'ids'           : idx,
            'input_ids'     : self.input_ids_tensor[idx],
            'attention_mask': self.input_mask_tensor[idx],
            'mask'          : self.predict_mask_tensor[idx],
        }
        return d


class BERTTaggerModel(BaseTorchSeqModel):
    def __init__(self,
                 model_name: Optional[str] = 'bert-base-uncased',
                 lr: Optional[float] = 2e-5,
                 l2: Optional[float] = 1e-6,
                 max_tokens: Optional[int] = 512,
                 batch_size: Optional[int] = 32,
                 real_batch_size: Optional[int] = 32,
                 test_batch_size: Optional[int] = 128,
                 n_steps: Optional[int] = 10000,
                 use_crf: Optional[bool] = False,
                 fine_tune_layers: Optional[int] = -1,
                 lr_crf: Optional[float] = 5e-5,
                 l2_crf: Optional[float] = 1e-8,
                 ):
        super().__init__()
        self.hyperparas = {
            'fine_tune_layers': fine_tune_layers,
            'model_name'      : model_name,
            'lr'              : lr,
            'l2'              : l2,
            'max_tokens'      : max_tokens,
            'batch_size'      : batch_size,
            'real_batch_size' : real_batch_size,
            'test_batch_size' : test_batch_size,
            'n_steps'         : n_steps,
            'use_crf'         : use_crf,
            'lr_crf'          : lr_crf,
            'l2_crf'          : l2_crf,
        }
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _init_valid_dataloader(self, dataset_valid: BaseSeqDataset) -> DataLoader:
        torch_dataset = BERTTorchSeqDataset(dataset_valid, self.tokenizer, 512, self.hyperparas['use_crf'])
        valid_dataloader = DataLoader(torch_dataset, batch_size=self.hyperparas['test_batch_size'], shuffle=False,
                                      collate_fn=collate_fn)
        return valid_dataloader

    def fit(self,
            dataset_train: BaseSeqDataset,
            y_train: Optional[List[List]] = None,
            dataset_valid: Optional[BaseSeqDataset] = None,
            y_valid: Optional[List[List]] = None,
            evaluation_step: Optional[int] = 50,
            metric: Optional[Union[str, Callable]] = 'f1_seq',
            strict: Optional[bool] = True,
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
        if hyperparas['real_batch_size'] == -1 or hyperparas['batch_size'] < hyperparas['real_batch_size']:
            hyperparas['real_batch_size'] = hyperparas['batch_size']
        accum_steps = hyperparas['batch_size'] // hyperparas['real_batch_size']

        n_steps = hyperparas['n_steps']
        torch_dataset = BERTTorchSeqDataset(dataset_train, self.tokenizer, self.hyperparas['max_tokens'],
                                            self.hyperparas['use_crf'], n_data=n_steps * hyperparas['batch_size'])
        train_dataloader = DataLoader(torch_dataset, batch_size=hyperparas['real_batch_size'], shuffle=True, collate_fn=collate_fn)

        if y_train is None:
            y_train = dataset_train.labels
        y_train = torch_dataset.prepare_labels(y_train).to(device)

        n_class = dataset_train.n_class
        model = BertSeqTagger(n_class, **hyperparas).to(device)
        self.model = model

        param_optimizer = list(model.named_parameters())
        crf_param = ['crf.transitions', ]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if n not in crf_param]},
            {'params'      : [p for n, p in param_optimizer if n in crf_param], 'lr': hyperparas['lr_crf'],
             'weight_decay': hyperparas['l2_crf']},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=hyperparas['lr'], weight_decay=hyperparas['l2'])

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=n_steps)

        valid_flag = self._init_valid_step(dataset_valid, y_valid, metric, strict, direction, patience, tolerance)

        history = {}
        last_step_log = {}
        try:
            with trange(n_steps, desc=f"[FINETUNE] {hyperparas['model_name']} Tagger", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for batch in train_dataloader:

                    batch_idx = batch['ids'].to(device)
                    batch_label = y_train[batch_idx]
                    loss = model.calculate_loss(batch, batch_label)
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
