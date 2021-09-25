import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import trange
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModel, AutoConfig

from ..backbone import WordSequence, BackBone, MultiCRF
from ..dataset.seqdataset import BaseSeqDataset
from ..seq_endmodel.bert_crf_model import BERTTaggerModel, BERTTorchSeqDataset
from ..seq_endmodel.lstm_crf_model import LSTMTaggerModel, LSTMTorchSeqDataset
from ..utils import construct_collate_fn_trunc_pad

logger = logging.getLogger(__name__)

collate_fn = construct_collate_fn_trunc_pad('mask')


def agg_labels(weak_labels, n_class):  # use MV (since no priors) to aggregate weak labels
    agg_weak_labels = []
    for i in range(len(weak_labels)):
        L = weak_labels[i]
        ni = len(L)
        Y_p = np.zeros((ni, n_class))
        for i in range(ni):
            counts = np.zeros(n_class)
            for j in range(L.shape[1]):
                if L[i, j] != 0:
                    counts[L[i, j]] += 1
            if counts.sum() == 0:
                counts[0] = 1
            Y_p[i, :] = counts
        Y_p /= Y_p.sum(axis=1, keepdims=True)
        Y_p = np.argmax(Y_p, axis=-1)
        agg_weak_labels.append(list(Y_p))
    return agg_weak_labels


class LSTMConNetModel(LSTMTaggerModel):
    def __init__(self,
                 n_steps_phase1: Optional[int] = 200,
                 n_steps: Optional[int] = 10000,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.hyperparas.update({
            'n_steps'       : n_steps,
            'n_steps_phase1': n_steps_phase1,
            'use_crf'       : True,
        })

    def fit(self,
            dataset_train: BaseSeqDataset,
            dataset_valid: Optional[BaseSeqDataset] = None,
            y_valid: Optional[List[List]] = None,
            word_embedding: Optional[np.ndarray] = None,
            char_embedding: Optional[np.ndarray] = None,
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
        self.hyperparas['use_crf'] = True
        hyperparas = self.hyperparas

        assert (word_embedding is None) or hyperparas['word_emb_dim'] == word_embedding.shape[1]
        assert (char_embedding is None) or hyperparas['char_emb_dim'] == char_embedding.shape[1]

        n_steps = hyperparas['n_steps']
        n_steps_phase1 = hyperparas['n_steps_phase1']
        n_steps_total = n_steps_phase1 + n_steps
        torch_dataset = LSTMTorchSeqDataset(dataset_train, n_data=n_steps_total * hyperparas['batch_size'])
        train_dataloader = DataLoader(torch_dataset, batch_size=hyperparas['batch_size'], shuffle=True, collate_fn=collate_fn)

        n_class = dataset_train.n_class
        n_source = dataset_train.n_lf
        n_data, seq_len = torch_dataset.word_seq_tensor.shape
        O_id = dataset_train.label2id['O']

        weak_labels_pad = np.ones((n_data, seq_len, n_source), dtype=int) * O_id
        for i, weak_labels_i in enumerate(dataset_train.weak_labels):
            ni = len(weak_labels_i)
            if ni > seq_len:
                weak_labels_pad[i] = weak_labels_i[:seq_len]
            else:
                weak_labels_pad[i, :ni] = weak_labels_i
        weak_labels = torch.LongTensor(weak_labels_pad).to(device)

        agg_weak_labels = agg_labels(dataset_train.weak_labels, n_class)
        agg_weak_labels_pad = np.ones((n_data, seq_len), dtype=int) * O_id
        for i, agg_weak_labels_i in enumerate(agg_weak_labels):
            ni = len(agg_weak_labels_i)
            if ni > seq_len:
                agg_weak_labels_pad[i] = agg_weak_labels_i[:seq_len]
            else:
                agg_weak_labels_pad[i, :ni] = agg_weak_labels_i
        agg_weak_labels = torch.LongTensor(agg_weak_labels_pad).to(device)

        word_vocab_size = len(dataset_train.word_dict)
        char_vocab_size = len(dataset_train.char_dict)
        model = LSTMConNet(
            n_class=n_class,
            n_source=n_source,
            word_vocab_size=word_vocab_size,
            char_vocab_size=char_vocab_size,
            word_embedding=word_embedding,
            char_embedding=char_embedding,
            **hyperparas).to(device)
        self.model = model

        optimizer = optim.Adam(model.parameters(), lr=hyperparas['lr'], weight_decay=hyperparas['l2'])

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=n_steps_total)

        valid_flag = self._init_valid_step(dataset_valid, y_valid, metric, strict, direction, patience, tolerance)

        history_phase1 = {}
        last_step_log = {'loss': -1}
        with trange(n_steps_phase1, desc="[TRAIN] LSTM-ConNet Phase 1", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
            model.train()
            step = 0
            for batch in train_dataloader:
                step += 1
                optimizer.zero_grad()

                batch_idx = batch['ids'].to(device)
                batch_weak_label = weak_labels[batch_idx]
                loss = model.calc_phase1_loss(batch, batch_weak_label)

                loss.backward()
                optimizer.step()
                scheduler.step()

                history_phase1[step] = {
                    'loss': loss.item(),
                }
                last_step_log.update(history_phase1[step])

                pbar.update()
                pbar.set_postfix(ordered_dict=last_step_log)

                if step >= n_steps_phase1:
                    break

        history_phase2 = {}
        last_step_log = {'loss': -1}
        with trange(n_steps, desc="[TRAIN] LSTM-ConNet Phase 2", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
            model.train()
            model.freeze_encoder()
            step = 0
            for batch in train_dataloader:
                step += 1
                optimizer.zero_grad()

                batch_idx = batch['ids'].to(device)
                batch_agg_weak_label = agg_weak_labels[batch_idx]
                loss = model.calc_phase2_loss(batch, batch_agg_weak_label)

                loss.backward()
                optimizer.step()
                scheduler.step()

                if valid_flag and step % evaluation_step == 0:
                    metric_value, early_stop_flag, info = self._valid_step(step)
                    if early_stop_flag:
                        logger.info(info)
                        break

                    history_phase2[step] = {
                        'loss'              : loss.item(),
                        f'val_{metric}'     : metric_value,
                        f'best_val_{metric}': self.best_metric_value,
                        'best_step'         : self.best_step,
                    }
                    last_step_log.update(history_phase2[step])

                last_step_log['loss'] = loss.item()
                pbar.update()
                pbar.set_postfix(ordered_dict=last_step_log)

                if step >= n_steps:
                    break

        self._finalize()

        history = {
            'phase1': history_phase1,
            'phase2': history_phase2,
        }

        return history


class BERTConNetModel(BERTTaggerModel):
    def __init__(self,
                 n_steps_phase1: Optional[int] = 200,
                 n_steps: Optional[int] = 10000,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.hyperparas.update({
            'n_steps'       : n_steps,
            'n_steps_phase1': n_steps_phase1,
            'use_crf'       : True,
        })

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
        self.hyperparas['use_crf'] = True
        hyperparas = self.hyperparas
        accum_steps = hyperparas['batch_size'] // hyperparas['real_batch_size']

        n_steps = hyperparas['n_steps']
        n_steps_phase1 = hyperparas['n_steps_phase1']
        n_steps_total = n_steps_phase1 + n_steps
        torch_dataset = BERTTorchSeqDataset(dataset_train, self.tokenizer, self.hyperparas['max_tokens'],
                                            self.hyperparas['use_crf'], n_data=n_steps_total * hyperparas['batch_size'])
        train_dataloader = DataLoader(torch_dataset, batch_size=hyperparas['real_batch_size'], shuffle=True,
                                      collate_fn=collate_fn)

        n_class = dataset_train.n_class
        n_source = dataset_train.n_lf
        n_data, seq_len = torch_dataset.input_ids_tensor.shape
        O_id = dataset_train.label2id['O']

        weak_labels_pad = np.ones((n_data, seq_len, n_source), dtype=int) * O_id
        for i, weak_labels_i in enumerate(dataset_train.weak_labels):
            ni = len(weak_labels_i)
            if ni > seq_len:
                weak_labels_pad[i] = weak_labels_i[:seq_len]
            else:
                weak_labels_pad[i, :ni] = weak_labels_i
        weak_labels = torch.LongTensor(weak_labels_pad).to(device)

        agg_weak_labels = agg_labels(dataset_train.weak_labels, n_class)
        agg_weak_labels_pad = np.ones((n_data, seq_len), dtype=int) * O_id
        for i, agg_weak_labels_i in enumerate(agg_weak_labels):
            ni = len(agg_weak_labels_i)
            if ni > seq_len:
                agg_weak_labels_pad[i] = agg_weak_labels_i[:seq_len]
            else:
                agg_weak_labels_pad[i, :ni] = agg_weak_labels_i
        agg_weak_labels = torch.LongTensor(agg_weak_labels_pad).to(device)

        model = BERTConNet(
            n_class=n_class,
            n_source=n_source,
            **hyperparas
        ).to(device)
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
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=n_steps_total)

        valid_flag = self._init_valid_step(dataset_valid, y_valid, metric, strict, direction, patience, tolerance)

        history_phase1 = {}
        last_step_log = {}
        with trange(n_steps_phase1, desc=f"[FINETUNE] {hyperparas['model_name']} BERT-ConNet Phase 1", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
            cnt = 0
            step = 0
            model.train()
            optimizer.zero_grad()
            for batch in train_dataloader:

                batch_idx = batch['ids'].to(device)
                batch_weak_label = weak_labels[batch_idx]
                loss = model.calc_phase1_loss(batch, batch_weak_label)

                loss.backward()
                cnt += 1

                if cnt % accum_steps == 0:
                    # Clip the norm of the gradients to 1.0.
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1

                    history_phase1[step] = {
                        'loss': loss.item(),
                    }
                    last_step_log.update(history_phase1[step])

                    pbar.update()
                    pbar.set_postfix(ordered_dict=last_step_log)

                    if step >= n_steps_phase1:
                        break

        history_phase2 = {}
        last_step_log = {}
        with trange(n_steps, desc=f"[FINETUNE] {hyperparas['model_name']} BERT-ConNet Phase 2", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
            cnt = 0
            step = 0
            model.train()
            model.freeze_encoder()
            optimizer.zero_grad()
            for batch in train_dataloader:

                batch_idx = batch['ids'].to(device)
                batch_agg_weak_label = agg_weak_labels[batch_idx]
                loss = model.calc_phase2_loss(batch, batch_agg_weak_label)

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

                        history_phase2[step] = {
                            'loss'              : loss.item(),
                            f'val_{metric}'     : metric_value,
                            f'best_val_{metric}': self.best_metric_value,
                            'best_step'         : self.best_step,
                        }
                        last_step_log.update(history_phase2[step])

                    last_step_log['loss'] = loss.item()
                    pbar.update()
                    pbar.set_postfix(ordered_dict=last_step_log)

                    if step >= n_steps:
                        break

        self._finalize()

        history = {
            'phase1': history_phase1,
            'phase2': history_phase2,
        }

        return history


class BaseConNet(BackBone, ABC):
    '''
    Connet Model for sequence tagging
    '''

    @abstractmethod
    def freeze_encoder(self):
        pass

    @abstractmethod
    def encode(self, batch):
        pass

    def forward(self, batch):
        # Only for inferencing
        device = self.get_device()
        mask = batch['mask'].to(device)

        feats, outs = self.encode(batch)
        seq_feature = self.get_feature(feats, mask)  # [B, 2 * hidden_dim]
        attn_weight = F.softmax(self.weight(seq_feature), -1)  # [B, n_source]
        _, tag_seq = self.crf(outs, mask, attn_weight)
        return tag_seq

    def calc_phase1_loss(self, batch, batch_weak_labels):
        '''decoupling phase, learning #LF transition matrix separately'''
        device = self.get_device()
        _, outs = self.encode(batch)
        mask = batch['mask'].bool().to(device)

        seq_len = mask.shape[1]
        batch_weak_labels = batch_weak_labels[:, :seq_len, :].to(device)

        total_loss = 0
        for i in range(self.n_source):
            total_loss += self.crf.neg_log_likelihood_loss(outs, mask, batch_weak_labels[:, :, i], i)
        total_loss = total_loss / (outs.shape[0] * self.n_source)
        return total_loss

    def calc_phase2_loss(self, batch, batch_agg_weak_labels):
        '''aggregation phase, learning #LF transition matrix separately'''
        device = self.get_device()
        feats, outs = self.encode(batch)
        mask = batch['mask'].to(device)

        seq_len = mask.shape[1]
        agg_weak_labels = batch_agg_weak_labels[:, :seq_len].to(device)

        seq_feature = self.get_feature(feats, mask)  # [B, 2 * hidden_dim]
        attn_weight = F.softmax(self.weight(seq_feature), -1)  # [B, n_source]
        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, agg_weak_labels, idx=None, attn_weight=attn_weight)
        total_loss = total_loss / outs.shape[0]
        return total_loss

    def get_feature(self, feats, mask):
        '''Get sentence-level representation, concatenating the embedding of the first token and the final token'''
        # feats [B, len, hidden_dim] -> [B, 2 * hidden_dim]
        batch_size = feats.size(0)
        start_feat = feats[:, 0, :]
        end_feat = feats[torch.arange(batch_size), torch.sum(mask.long(), dim=1) - 1]
        return torch.cat([start_feat, end_feat], -1)


class LSTMConNet(BaseConNet):
    '''
    Connet Model for sequence tagging
    '''

    def __init__(self, n_class,
                 n_source,
                 word_vocab_size,
                 char_vocab_size,
                 dropout,
                 word_embedding,
                 word_emb_dim,
                 word_hidden_dim,
                 word_feature_extractor,
                 n_word_hidden_layer,
                 use_char,
                 char_embedding,
                 char_emb_dim,
                 char_hidden_dim,
                 char_feature_extractor,
                 **kwargs):
        super(LSTMConNet, self).__init__(n_class=n_class)
        self.n_source = n_source

        self.model = WordSequence(
            word_vocab_size=word_vocab_size,
            char_vocab_size=char_vocab_size,
            dropout=dropout,
            word_embedding=word_embedding,
            word_emb_dim=word_emb_dim,
            word_hidden_dim=word_hidden_dim,
            word_feature_extractor=word_feature_extractor,
            n_word_hidden_layer=n_word_hidden_layer,
            use_char=use_char,
            char_embedding=char_embedding,
            char_emb_dim=char_emb_dim,
            char_hidden_dim=char_hidden_dim,
            char_feature_extractor=char_feature_extractor
        )
        self.classifier = nn.Linear(word_hidden_dim, self.n_class + 2)
        self.weight = nn.Linear(2 * word_hidden_dim, self.n_source)

        self.crf = MultiCRF(n_class=self.n_class, n_source=self.n_source)

    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False

    def encode(self, batch):
        '''decoupling phase, learning #LF transition matrix separately'''
        device = self.get_device()
        word_inputs = batch['word'].to(device)
        word_seq_lengths = batch['word_length']
        char_inputs = batch['char'].to(device)
        char_seq_lengths = batch['char_length']
        char_inputs = char_inputs.flatten(0, 1)
        char_seq_lengths = char_seq_lengths.flatten()

        feats = self.model(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths)
        outs = self.classifier(feats)
        return feats, outs


class BERTConNet(BaseConNet):
    '''
    Connet Model for sequence tagging
    '''

    def __init__(self, n_class, n_source, model_name='bert-base-cased', fine_tune_layers=-1, **kwargs):
        super(BERTConNet, self).__init__(n_class=n_class)
        self.n_source = n_source
        self.model_name = model_name
        config = AutoConfig.from_pretrained(self.model_name, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(self.model_name, config=config)
        self.config = config

        if fine_tune_layers >= 0:
            for param in self.model.base_model.embeddings.parameters(): param.requires_grad = False
            if fine_tune_layers > 0:
                n_layers = len(self.model.base_model.encoder.layer)
                for layer in self.model.base_model.encoder.layer[:n_layers - fine_tune_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # BERT have a smaller dropout rate, use 0.5 will hurt the performance
        self.classifier = nn.Linear(config.hidden_size, self.n_class + 2)  # consider <START> and <END> token
        self.weight = nn.Linear(2 * config.hidden_size, self.n_source)
        self.crf = MultiCRF(n_class=self.n_class, n_source=self.n_source)

    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, batch):
        '''decoupling phase, learning #LF transition matrix separately'''
        device = self.get_device()
        outputs = self.model(input_ids=batch["input_ids"].to(device), attention_mask=batch['attention_mask'].to(device))
        feats = outputs.last_hidden_state
        outs = self.classifier(self.dropout(outputs.last_hidden_state))
        return feats, outs
