import json
from pathlib import Path
from typing import Any, List, Optional, Union

import os
import numpy as np
import torch
from torchvision.datasets.folder import pil_loader

from .basedataset import BaseDataset
from .utils import bag_of_words_extractor, tf_idf_extractor, sentence_transformer_extractor, \
    bert_text_extractor, bert_relation_extractor, image_feature_extractor


class NumericDataset(BaseDataset):
    """Data class for numeric dataset."""

    def extract_feature_(self,
                         extract_fn: str,
                         return_extractor: bool,
                         **kwargs: Any):
        """Method for extracting features for NumericDataset: convert list of list to np.array.

        Parameters
        ----------
        """
        self.features = np.array(list(map(lambda x: x['feature'], self.examples)), dtype=np.float32)

        if return_extractor:
            return lambda y: np.array(list(map(lambda x: x['feature'], y)), dtype=np.float32)


class TextDataset(BaseDataset):
    """Data class for text classification dataset."""


    def extract_feature_(self,
                         extract_fn: str,
                         return_extractor: bool,
                         device: torch.device = None,
                         model_name: Optional[str] = 'bert-base-cased',
                         feature: Optional[str] = 'cls',
                         **kwargs: Any):
        """Method for extracting features for TextDataset.

        Parameters
        ----------
        extract_fn
            str with values in {'bow', 'tfidf', 'sentence_transformer', 'bert'} or customized Callable function.
        return_extractor
            Whether to return feature extractor.
        """

        if extract_fn == 'bow':
            data, extractor = bag_of_words_extractor(self.examples, **kwargs)
        elif extract_fn == 'tfidf':
            data, extractor = tf_idf_extractor(self.examples, **kwargs)
        elif extract_fn == 'sentence_transformer':
            data, extractor = sentence_transformer_extractor(self.examples,
                                                             device=device,
                                                             model_name=model_name,
                                                             **kwargs)
        elif extract_fn == 'bert':
            data, extractor = bert_text_extractor(self.examples,
                                                  device=device,
                                                  model_name=model_name,
                                                  feature=feature,
                                                  **kwargs)
        else:
            raise NotImplementedError(f'feature extraction method {extract_fn} is not supported!')

        self.features = data

        if return_extractor:
            return extractor
    


class RelationDataset(BaseDataset):
    """Data class for relation dataset."""

    def extract_feature_(self,
                         extract_fn: str,
                         return_extractor: bool,
                         device: torch.device = None,
                         model_name: Optional[str] = 'bert-base-cased',
                         feature: Optional[str] = 'cat',
                         **kwargs: Any):
        """Method for extracting features for TextDataset.

        Parameters
        ----------
        extract_fn
            str with values in {'bert'} or customized Callable function.
        return_extractor
            Whether to return feature extractor.
        """

        if extract_fn == 'bert':
            data, extractor = bert_relation_extractor(self.examples,
                                                      device=device,
                                                      model_name=model_name,
                                                      feature=feature,
                                                      **kwargs)
        else:
            raise NotImplementedError(f'feature extraction method {extract_fn} is not supported!')

        self.features = data

        if return_extractor:
            return extractor


class ImageDataset(BaseDataset):
    """Data class for image dataset."""

    def __init__(self,
                 path: Union[str, Path] = None,
                 split: Optional[str] = None,
                 image_root_path: Optional[str] = None,
                 preload_image: Optional[bool] = True,
                 feature_cache_name: Optional[str] = None,
                 **kwargs: Any) -> None:
        self.image_root_path = image_root_path
        self.preload_image = preload_image
        super(ImageDataset, self).__init__(path=path, split=split, feature_cache_name=feature_cache_name, **kwargs)

    def load(self, path: Union[str, Path], split: str):
        super(ImageDataset, self).load(path=path, split=split)

        with open(path / 'meta.json', 'r', encoding='utf-8') as f:
            meta_dict = json.load(f)
        self.image_input_size = meta_dict['input_size']

        imgs, np_imgs = [], []
        image_root_path = Path(self.image_root_path)
        for d in self.examples:
            d['image_path'] = str(image_root_path / d['image_path'])
            img = pil_loader(d['image_path']).resize(self.image_input_size)
            imgs.append(img)
            np_imgs.append(np.asarray(img, dtype='uint8'))
        np_imgs = np.asarray(np_imgs) / 255.0
        self.image_mean = np.mean(np_imgs, axis=(0, 1, 2))
        self.image_std = np.std(np_imgs, axis=(0, 1, 2))

        if self.preload_image:
            self.images = imgs

        return self

    def create_subset(self, idx: List[int]):
        dataset = super(ImageDataset, self).create_subset(idx=idx)

        if self.preload_image:
            dataset.images = [self.images[i] for i in idx]

        dataset.image_root_path = self.image_root_path
        dataset.preload_image = self.preload_image
        dataset.image_input_size = self.image_input_size
        dataset.image_mean = self.image_mean
        dataset.image_std = self.image_std

        return dataset

    def extract_feature_(self,
                         extract_fn: str,
                         return_extractor: bool,
                         device: torch.device = None,
                         model_name: Optional[str] = 'resnet18',
                         **kwargs: Any):
        if extract_fn == 'pretrain':
            data, extractor = image_feature_extractor(self.examples,
                                                      device=device,
                                                      model_name=model_name,
                                                      **kwargs)
        else:
            raise NotImplementedError(f'feature extraction method {extract_fn} is not supported!')

        self.features = data

        if return_extractor:
            return extractor
