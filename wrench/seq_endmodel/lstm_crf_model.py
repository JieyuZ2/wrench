import logging
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union, Callable

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange
from transformers import get_linear_schedule_with_warmup

from ..backbone import LSTMSeqTagger
from ..basemodel import BaseTorchSeqModel
from ..dataset.seqdataset import BaseSeqDataset
from ..utils import construct_collate_fn_trunc_pad

logger = logging.getLogger(__name__)

collate_fn = construct_collate_fn_trunc_pad('mask')


def encode(examples, word_dict: Dict, char_dict: Dict):
    word_ids_list = []
    word_unk_id = word_dict["<UNK>"]
    char_ids_list = []
    char_unk_id = char_dict["<UNK>"]
    for example in examples:
        word_ids = []
        char_ids = []
        for word in example['text']:
            word_ids.append(word_dict.get(word.lower(), word_unk_id))
            char_ids.append([char_dict.get(char, char_unk_id) for char in word])
        word_ids_list.append(word_ids)
        char_ids_list.append(char_ids)
    return word_ids_list, char_ids_list


def batchify(word_ids_list, char_ids_list, word_dict: Dict, char_dict: Dict):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    word_pad_id = word_dict["<PAD>"]
    char_pad_id = char_dict["<PAD>"]
    batch_size = len(word_ids_list)

    word_seq_lengths = list(map(len, word_ids_list))
    max_seq_len = max(word_seq_lengths)

    word_seq_tensor = word_pad_id * np.ones((batch_size, max_seq_len))
    for idx, (seq, seqlen) in enumerate(zip(word_ids_list, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = seq

    pad_chars = [char_ids + [[char_pad_id]] * (max_seq_len - len(char_ids)) for char_ids in char_ids_list]
    char_seq_lengths = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, char_seq_lengths))
    char_seq_tensor = char_pad_id * np.ones((batch_size, max_seq_len, max_word_len))
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = word

    word_seq_tensor = torch.LongTensor(word_seq_tensor)
    word_seq_lengths = torch.LongTensor(word_seq_lengths)

    char_seq_tensor = torch.LongTensor(char_seq_tensor)
    char_seq_lengths = torch.LongTensor(char_seq_lengths)

    mask = torch.BoolTensor(word_seq_tensor != word_pad_id)

    return word_seq_tensor, word_seq_lengths, char_seq_tensor, char_seq_lengths, mask, max_seq_len


class LSTMTorchSeqDataset(Dataset):
    def __init__(self, dataset: BaseSeqDataset, n_data: Optional[int] = 0):
        self.id2label = deepcopy(dataset.id2label)
        self.label2id = deepcopy(dataset.label2id)
        self.n_class = len(self.id2label)
        self.seq_len = list(map(len, map(lambda x: x["text"], dataset.examples)))

        word_ids_list, char_ids_list = encode(dataset.examples, word_dict=dataset.word_dict, char_dict=dataset.char_dict)
        word_seq_tensor, word_seq_lengths, char_seq_tensor, char_seq_lengths, mask, max_seq_len \
            = batchify(word_ids_list, char_ids_list, dataset.word_dict, dataset.char_dict)

        self.word_seq_tensor = word_seq_tensor
        self.word_seq_lengths = word_seq_lengths
        self.char_seq_tensor = char_seq_tensor
        self.char_seq_lengths = char_seq_lengths
        self.mask = mask
        self.max_seq_len = max_seq_len

        n_data_ = len(mask)
        self.n_data_ = n_data_
        if n_data > 0:
            self.n_data = math.ceil(n_data / n_data_) * n_data_
        else:
            self.n_data = n_data_

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        idx = idx % self.n_data_
        d = {
            'ids'        : idx,
            'word'       : self.word_seq_tensor[idx],
            'word_length': self.word_seq_lengths[idx],
            'char'       : self.char_seq_tensor[idx],
            'char_length': self.char_seq_lengths[idx],
            "mask"       : self.mask[idx]
        }
        return d


class LSTMTaggerModel(BaseTorchSeqModel):
    def __init__(self,
                 lr: Optional[float] = 1e-2,
                 l2: Optional[float] = 1e-8,
                 batch_size: Optional[int] = 64,
                 test_batch_size: Optional[int] = 512,
                 n_steps: Optional[int] = 10000,
                 use_crf: Optional[bool] = True,
                 dropout: Optional[float] = 0.5,

                 word_emb_dim: Optional[int] = 100,
                 word_hidden_dim: Optional[int] = 200,
                 word_feature_extractor: Optional[str] = 'LSTM',
                 n_word_hidden_layer: Optional[int] = 1,

                 use_char: Optional[bool] = True,
                 char_emb_dim: Optional[int] = 30,
                 char_hidden_dim: Optional[int] = 50,
                 char_feature_extractor: Optional[str] = 'CNN',  # 'CNN-LSTM'

                 ):
        super().__init__()
        self.hyperparas = {
            'lr'                    : lr,
            'l2'                    : l2,
            'n_steps'               : n_steps,
            'use_crf'               : use_crf,
            'batch_size'            : batch_size,
            'test_batch_size'       : test_batch_size,
            'dropout'               : dropout,

            'word_emb_dim'          : word_emb_dim,
            'word_hidden_dim'       : word_hidden_dim,
            'word_feature_extractor': word_feature_extractor,
            'n_word_hidden_layer'   : n_word_hidden_layer,

            'use_char'              : use_char,
            'char_emb_dim'          : char_emb_dim,
            'char_hidden_dim'       : char_hidden_dim,
            'char_feature_extractor': char_feature_extractor,

        }
        self.model = None

    def _init_valid_dataloader(self, dataset_valid: BaseSeqDataset) -> DataLoader:
        torch_dataset = LSTMTorchSeqDataset(dataset_valid)
        valid_dataloader = DataLoader(torch_dataset, batch_size=self.hyperparas['test_batch_size'], shuffle=False, collate_fn=collate_fn)
        return valid_dataloader

    def fit(self,
            dataset_train: BaseSeqDataset,
            y_train: Optional[List[List]] = None,
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
        hyperparas = self.hyperparas

        assert (word_embedding is None) or hyperparas['word_emb_dim'] == word_embedding.shape[1]
        assert (char_embedding is None) or hyperparas['char_emb_dim'] == char_embedding.shape[1]

        n_steps = hyperparas['n_steps']
        torch_dataset = LSTMTorchSeqDataset(dataset_train, n_data=n_steps * hyperparas['batch_size'])
        train_dataloader = DataLoader(torch_dataset, batch_size=hyperparas['batch_size'], shuffle=True, collate_fn=collate_fn)

        if y_train is None:
            y_train = dataset_train.labels

        y_train_pad = np.zeros((len(y_train), torch_dataset.max_seq_len)) * dataset_train.word_dict[dataset_train.PAD]
        for i, y_train_i in enumerate(y_train):
            y_train_pad[i, :len(y_train_i)] = y_train_i
        y_train = torch.LongTensor(y_train_pad).to(device)

        word_vocab_size = len(dataset_train.word_dict)
        char_vocab_size = len(dataset_train.char_dict)
        n_class = dataset_train.n_class
        model = LSTMSeqTagger(
            n_class=n_class,
            word_vocab_size=word_vocab_size,
            char_vocab_size=char_vocab_size,
            word_embedding=word_embedding,
            char_embedding=char_embedding,
            **hyperparas).to(device)
        self.model = model

        optimizer = optim.Adam(model.parameters(), lr=hyperparas['lr'], weight_decay=hyperparas['l2'])

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=n_steps)

        valid_flag = self._init_valid_step(dataset_valid, y_valid, metric, strict, direction, patience, tolerance)

        history = {}
        last_step_log = {}
        try:
            with trange(n_steps, desc="[TRAIN] LSTM Tagger", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                model.train()
                step = 0
                for batch in train_dataloader:
                    step += 1
                    optimizer.zero_grad()

                    batch_idx = batch['ids'].to(device)
                    batch_label = y_train[batch_idx]
                    loss = model.calculate_loss(batch, batch_label)

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
