import json
import logging
import pickle
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def entity_to_bio_labels(entities: List[str]):
    bio_labels = ["O"] + ["%s-%s" % (bi, label) for label in entities for bi in "BI"]
    return bio_labels


class BaseSeqDataset:
    """Abstract data class."""

    def __init__(self,
                 path: Union[str, Path] = None,
                 split: Optional[str] = None,
                 start_tag: Optional[str] = None,
                 stop_tag: Optional[str] = None,
                 pad_token: Optional[str] = None,
                 unk_token: Optional[str] = None,
                 **kwargs: Any) -> None:
        self.ids: List[List] = []
        self.labels: List[List] = []
        self.examples: List[List] = []
        self.weak_labels: List[List[List]] = []  # N * M * L

        self.START_TAG = start_tag or "<START>"
        self.STOP_TAG = stop_tag or "<STOP>"
        self.PAD = pad_token or "<PAD>"
        self.UNK = unk_token or "<UNK>"

        self.id2label = None
        self.label2id = None
        self.entity_types = None

        self.split = split
        self.path = path
        if path is not None and split is not None:
            self.load(path=path, split=split)
            self.n_class = len(self.id2label)
            self.n_lf = len(self.weak_labels[0][0])

    def __len__(self):
        return len(self.ids)

    def load(self, path: Union[str, Path], split: str):
        """Method for loading data given the split.

        Parameters
        ----------
        split
            A str with values in {"train", "valid", "test", None}. If None, then do not load any data.
        Returns
        -------
        self
        """

        assert split in ["train", "valid", "test"], 'Parameter "split" must be in ["train", "valid", "test", None]'

        path = Path(path)

        self.split = split
        self.path = path

        data_path = path / f'{split}.json'
        with open(data_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)

        # Load meta if exist
        with open(path / 'meta.json', 'r', encoding='utf-8') as f:
            meta_dict = json.load(f)

        bio_labels = entity_to_bio_labels(meta_dict['entity_types'])
        label_to_id = {lb: i for i, lb in enumerate(bio_labels)}
        np_map = np.vectorize(lambda lb: label_to_id[lb])

        if 'lf_rec' in meta_dict.keys():
            lf_rec_ids = [meta_dict['lf'].index(lf) for lf in meta_dict['lf_rec']]
        else:
            lf_rec_ids = list(range(meta_dict['num_lf']))

        sentence_list = list()
        label_list = list()
        weak_label_list = list()
        idx_list = list()

        for i, data in tqdm(data_dict.items()):
            idx_list.append(i)
            sentence_list.append(data['data'])
            label_list.append(np_map(data['label']))
            weak_lbs = np.asarray(data['weak_labels'])[:, lf_rec_ids]
            weak_lbs = np_map(weak_lbs)
            weak_label_list.append(weak_lbs)

        self.ids = idx_list
        self.labels = label_list
        self.weak_labels = weak_label_list
        self.examples = sentence_list
        self.label2id = label_to_id
        self.id2label = bio_labels
        self.entity_types = meta_dict['entity_types']

        return self

    def flatten(self):
        L = []  # weak labels,
        Y = []  # labels,
        indexes = [0]
        for i in range(len(self)):
            L += list(self.weak_labels[i])  # size: (#tokens) * (#LF)
            Y += list(self.labels[i])  # size (# tokens)
            indexes.append(len(self.labels[i]))
        indexes = np.cumsum(indexes)
        return np.array(L), np.array(Y), indexes

    def load_embed_dict(self,
                        load_word_dict_path: Optional[Union[str, Path]] = None,
                        load_char_dict_path: Optional[Union[str, Path]] = None,
                        word_embed_dict: Optional[dict] = None):
        if load_word_dict_path is not None:
            self.word_dict = pickle.load(open(load_word_dict_path, 'wr'))
        else:
            self.word_dict = word_embed_dict

        if load_char_dict_path is not None:
            self.char_dict = pickle.load(open(load_char_dict_path, 'wr'))
        else:
            self.char_dict = {}
            self.char = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
            for i, c in enumerate(self.char):
                self.char_dict[c] = i
            self.char_dict[self.PAD] = len(self.char_dict)
            self.char_dict[self.UNK] = len(self.char_dict)

    def save_embed_dict(self, save_word_dict_path: Optional[str] = None, save_char_dict_path: Optional[str] = None):
        pickle.dump(self.word_dict, open(save_word_dict_path, 'wb'))
        pickle.dump(self.char_dict, open(save_char_dict_path, 'wb'))

    def load_bert_embed(self, model_name="bert-base-uncased", device: Optional[torch.device] = None, load_path=None):

        if load_path is not None:
            logger.info(f'load bert embedding from {load_path}')
            self.bert_embeddings = pickle.load(open(load_path, 'rb'))
            return self.bert_embeddings

        corpus = list(map(lambda x: x['text'], self.examples))
        self.bert_embeddings = build_bert_embeddings(corpus, model_name, device)

        return self.bert_embeddings

    def save_bert_embed(self, save_path):
        logger.info(f'save bert embedding to {save_path}')
        pickle.dump(self.bert_embeddings, open(save_path, 'wb'))


import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from tokenizations import get_alignments
from transformers import AutoTokenizer, AutoModel


def build_bert_embeddings(corpus, bert_model_name, device):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    model = AutoModel.from_pretrained(bert_model_name).to(device)

    standarized_sents = list()
    o2n_map = list()
    n = 0

    # update input sentences so that every sentence has BERT length < 510
    for i, sents in enumerate(corpus):
        sent_str = ' '.join(sents)
        len_bert_tokens = len(tokenizer.tokenize(sent_str))

        # Deal with sentences that are longer than 512 BERT tokens
        if len_bert_tokens >= 510:
            sents_list = [sents]
            bert_length_list = [len(tokenizer.tokenize(' '.join(t))) for t in sents_list]
            while (np.asarray(bert_length_list) >= 510).any():
                splitted_sents_list = list()
                for tokens, bert_len in zip(sents_list, bert_length_list):

                    if bert_len < 510:
                        splitted_sents_list.append(tokens)
                        continue

                    sent_str = ' '.join(tokens)
                    splitted_sents = sent_tokenize(sent_str)

                    sent_lens = list()
                    for st in splitted_sents:
                        sent_lens.append(len(word_tokenize(st)))
                    ends = [np.sum(sent_lens[:i]) for i in range(1, len(sent_lens) + 1)]

                    nearest_end_idx = np.argmin((np.array(ends) - len(tokens) / 2) ** 2)
                    split_1 = tokens[:ends[nearest_end_idx]]
                    split_2 = tokens[ends[nearest_end_idx]:]
                    splitted_sents_list.append(split_1)
                    splitted_sents_list.append(split_2)
                sents_list = splitted_sents_list
                bert_length_list = [len(tokenizer.tokenize(' '.join(t))) for t in sents_list]
            n_splits = len(sents_list)
            standarized_sents += sents_list

            o2n_map.append(list(range(n, n + n_splits)))
            n += n_splits

        else:
            standarized_sents.append(sents)
            o2n_map.append([n])
            n += 1

    embs = list()
    for i, sent in enumerate(tqdm(standarized_sents, desc='extracting bert embedding...')):

        joint_sent = ' '.join(sent)
        bert_tokens = tokenizer.tokenize(joint_sent)

        input_ids = torch.tensor([tokenizer.encode(joint_sent, add_special_tokens=True)], device=device)
        # calculate BERT last layer embeddings
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0].squeeze(0).cpu()
            trunc_hidden_states = last_hidden_states[1:-1, :]

        ori2bert, bert2ori = get_alignments(sent, bert_tokens)  # there is an error that get_alignments func. cannot recognize [UNK] well

        emb_list = list()
        for j, idx in enumerate(ori2bert):
            if idx == []:
                if j == 0:
                    idx = [0]
                else:
                    idx = [max(ori2bert[j - 1])]
                emb = trunc_hidden_states[idx, :]
            else:
                emb = trunc_hidden_states[idx, :]
            emb_list.append(emb.mean(dim=0))

        bert_emb = torch.stack(emb_list)
        embs.append(bert_emb.cpu().detach())

    # Combine embeddings so that the embedding lengths equal to the lengths of the original sentences
    combined_embs = list()
    for o2n in o2n_map:
        if len(o2n) == 1:
            combined_embs.append(embs[o2n[0]].numpy())
        else:
            cat_emb = torch.cat([embs[ii] for ii in o2n], dim=0)
            combined_embs.append(cat_emb.numpy())

    return combined_embs
