from typing import Any, Optional

import numpy as np
import torch

from .basedataset import BaseDataset
from .utils import bag_of_words_extractor, tf_idf_extractor, sentence_transformer_extractor, \
    bert_text_extractor, bert_relation_extractor


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
