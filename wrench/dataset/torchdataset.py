import math
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader

from ..dataset import BaseDataset, TextDataset, RelationDataset, ImageDataset


def sample_batch(loader):
    while True:
        for batch in loader:
            yield batch


class TorchDataset(Dataset):
    def __init__(self, dataset: BaseDataset, n_data: Optional[int] = 0):
        self.features = dataset.features
        self.labels = dataset.labels
        self.weak_labels = np.array(dataset.weak_labels, dtype=np.float32)
        self.data = dataset.examples
        n_data_ = len(self.data)
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
            'labels'     : self.labels[idx],
            'weak_labels': self.weak_labels[idx],
            'data'       : self.data[idx],
        }
        if self.features is not None:
            d['features'] = self.features[idx]
        return d


class ImageTorchDataset(TorchDataset):
    def __init__(self, dataset: ImageDataset, n_data: Optional[int] = 0):
        super(ImageTorchDataset, self).__init__(dataset, n_data)
        self.preload_image = dataset.preload_image
        if self.preload_image:
            self.images = dataset.images
        self.input_size = dataset.image_input_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset.image_mean, std=dataset.image_std)
        ])

    def __getitem__(self, idx):
        idx = idx % self.n_data_
        if self.preload_image:
            img = self.images[idx]
        else:
            img = pil_loader(self.data[idx]['image_path']).resize(self.input_size)
        img = self.transform(img)
        d = {
            'ids'        : idx,
            'labels'     : self.labels[idx],
            'weak_labels': self.weak_labels[idx],
            'image'      : img,
        }
        if self.features is not None:
            d['features'] = self.features[idx]
        return d


class BERTTorchDataset(TorchDataset):
    def __init__(self,
                 dataset: BaseDataset,
                 tokenizer,
                 max_seq_length: Optional[int] = 512,
                 n_data: Optional[int] = 0,
                 return_features: Optional[bool] = False,
                 return_weak_labels: Optional[bool] = False,
                 return_labels: Optional[bool] = False,
                 ):
        super(BERTTorchDataset, self).__init__(dataset, n_data)
        self.return_features = return_features
        self.return_weak_labels = return_weak_labels
        self.return_labels = return_labels

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __getitem__(self, idx):
        idx = idx % self.n_data_
        d = self.getitem_(idx)
        if self.return_features:
            d['features'] = self.features[idx]
        if self.return_weak_labels:
            d['weak_labels'] = self.weak_labels[idx]
        if self.return_labels:
            d['labels'] = self.labels[idx]
        return d

    @abstractmethod
    def getitem_(self, idx):
        raise NotImplementedError


class BERTTorchTextClassDataset(BERTTorchDataset):
    def __init__(self,
                 dataset: TextDataset,
                 tokenizer,
                 max_seq_length: Optional[int] = 512,
                 n_data: Optional[int] = 0,
                 return_features: Optional[bool] = False,
                 return_weak_labels: Optional[bool] = False,
                 return_labels: Optional[bool] = False,
                 ):
        super(BERTTorchTextClassDataset, self).__init__(
            dataset,
            tokenizer,
            max_seq_length,
            n_data,
            return_features,
            return_weak_labels,
            return_labels,
        )
        corpus = list(map(lambda x: x["text"], dataset.examples))
        input_ids_tensor, input_mask_tensor = self.convert_corpus_to_tensor(corpus)

        self.input_ids_tensor = input_ids_tensor
        self.input_mask_tensor = input_mask_tensor

    def convert_corpus_to_tensor(self, corpus):
        outputs = self.tokenizer(corpus, return_token_type_ids=False, return_attention_mask=True, padding=True,
                                 return_tensors='pt', max_length=self.max_seq_length, truncation=True)
        input_ids_tensor = outputs['input_ids']
        input_mask_tensor = outputs['attention_mask']

        max_seq_length = input_mask_tensor.sum(dim=1).max()
        max_seq_length = min(max_seq_length, self.max_seq_length)

        input_ids_tensor = input_ids_tensor[:, :max_seq_length]
        input_mask_tensor = input_mask_tensor[:, :max_seq_length]

        return input_ids_tensor, input_mask_tensor

    def getitem_(self, idx):
        d = {
            'ids'      : idx,
            'input_ids': self.input_ids_tensor[idx],
            'mask'     : self.input_mask_tensor[idx],
        }
        return d


class BERTTorchRelationClassDataset(BERTTorchDataset):
    def __init__(self,
                 dataset: RelationDataset,
                 tokenizer,
                 max_seq_length: Optional[int] = 512,
                 n_data: Optional[int] = 0,
                 return_features: Optional[bool] = False,
                 return_weak_labels: Optional[bool] = False,
                 return_labels: Optional[bool] = False,
                 ):
        super(BERTTorchRelationClassDataset, self).__init__(
            dataset,
            tokenizer,
            max_seq_length,
            n_data,
            return_features,
            return_weak_labels,
            return_labels,
        )
        input_ids_tensor, input_mask_tensor, e1_mask_tensor, e2_mask_tensor = self.convert_corpus_to_tensor(dataset.examples)

        self.input_ids_tensor = input_ids_tensor
        self.input_mask_tensor = input_mask_tensor
        self.e1_mask_tensor = e1_mask_tensor
        self.e2_mask_tensor = e2_mask_tensor

    def convert_corpus_to_tensor(self, examples):
        max_seq_length = self.max_seq_length
        tokens_l, e1s_l, e1n_l, e2s_l, e2n_l = [], [], [], [], []
        for i, item in enumerate(examples):
            sentence = item['text']

            span1s, span1n, span2s, span2n = item['span1'][0], item['span1'][1], item['span2'][0], item['span2'][1]

            e1_first = span1s < span2s
            if e1_first:
                left_text = sentence[:span1s]
                between_text = sentence[span1n:span2s]
                right_text = sentence[span2n:]
            else:
                left_text = sentence[:span2s]
                between_text = sentence[span2n:span1s]
                right_text = sentence[span1n:]
            left_tkns = self.tokenizer.tokenize(left_text)
            between_tkns = self.tokenizer.tokenize(between_text)
            right_tkns = self.tokenizer.tokenize(right_text)
            e1_tkns = self.tokenizer.tokenize(item['entity1'])
            e2_tkns = self.tokenizer.tokenize(item['entity2'])

            if e1_first:
                tokens = ["[CLS]"] + left_tkns + ["$"] + e1_tkns + ["$"] + between_tkns + ["#"] + e2_tkns + [
                    "#"] + right_tkns + ["[SEP]"]
                e1s = len(left_tkns) + 1  # inclusive
                e1n = e1s + len(e1_tkns) + 2  # exclusive
                e2s = e1n + len(between_tkns)
                e2n = e2s + len(e2_tkns) + 2
                end = e2n
            else:
                tokens = ["[CLS]"] + left_tkns + ["#"] + e2_tkns + ["#"] + between_tkns + ["$"] + e1_tkns + [
                    "$"] + right_tkns + ["[SEP]"]
                e2s = len(left_tkns) + 1  # inclusive
                e2n = e2s + len(e2_tkns) + 2  # exclusive
                e1s = e2n + len(between_tkns)
                e1n = e1s + len(e1_tkns) + 2
                end = e1n

            if len(tokens) > max_seq_length:
                if end >= max_seq_length:
                    len_truncated = len(between_tkns) + len(e1_tkns) + len(e2_tkns) + 6
                    if len_truncated > max_seq_length:
                        diff = len_truncated - max_seq_length
                        len_between = len(between_tkns)
                        between_tkns = between_tkns[:(len_between - diff) // 2] + between_tkns[(len_between - diff) // 2 + diff:]
                    if e1_first:
                        truncated = ["[CLS]"] + ["$"] + e1_tkns + ["$"] + between_tkns + ["#"] + e2_tkns + ["#"] + [
                            "[SEP]"]
                        e1s = 1  # inclusive
                        e1n = e1s + len(e1_tkns) + 2  # exclusive
                        e2s = e1n + len(between_tkns)
                        e2n = e2s + len(e2_tkns) + 2
                    else:
                        truncated = ["[CLS]"] + ["#"] + e2_tkns + ["#"] + between_tkns + ["$"] + e1_tkns + ["$"] + [
                            "[SEP]"]
                        e2s = 1  # inclusive
                        e2n = e2s + len(e2_tkns) + 2  # exclusive
                        e1s = e2n + len(between_tkns)
                        e1n = e1s + len(e1_tkns) + 2
                    tokens = truncated
                    assert len(tokens) <= max_seq_length
                else:
                    tokens = tokens[:max_seq_length]

            assert e1_tkns == tokens[e1s + 1:e1n - 1]
            assert e2_tkns == tokens[e2s + 1:e2n - 1]

            e1s_l.append(e1s)
            e1n_l.append(e1n)
            e2s_l.append(e2s)
            e2n_l.append(e2n)
            tokens_l.append(self.tokenizer.convert_tokens_to_ids(tokens))

        max_len = max(list(map(len, tokens_l)))
        input_ids = torch.LongTensor([t + [self.tokenizer.pad_token_id] * (max_len - len(t)) for t in tokens_l])
        e1_mask = torch.zeros_like(input_ids)
        e2_mask = torch.zeros_like(input_ids)
        for i in range(len(examples)):
            e1_mask[i, e1s_l[i]:e1n_l[i]] = 1
            e2_mask[i, e2s_l[i]:e2n_l[i]] = 1
        input_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, input_mask, e1_mask, e2_mask

    def getitem_(self, idx):
        d = {
            'ids'      : idx,
            'input_ids': self.input_ids_tensor[idx],
            'mask'     : self.input_mask_tensor[idx],
            'e1_mask'  : self.e1_mask_tensor[idx],
            'e2_mask'  : self.e2_mask_tensor[idx],
        }
        return d
