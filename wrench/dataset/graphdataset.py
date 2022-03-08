import json
from pathlib import Path
from typing import Any, List, Optional, Union

import os
import numpy as np
import torch
import logging
from tqdm.auto import tqdm
from torchvision.datasets.folder import pil_loader
from .dataset import NumericDataset, TextDataset
from .basedataset import BaseDataset
from .utils import bag_of_words_extractor, tf_idf_extractor, sentence_transformer_extractor, \
    bert_text_extractor, bert_relation_extractor, image_feature_extractor


logger = logging.getLogger(__name__)
class GraphDataset(BaseDataset):
    def __init__(self,
                 path: str = None,
                 split: Optional[str] = None,
                 feature_cache_name: Optional[str] = None,
                 **kwargs: Any) -> None:
        import dgl
        super().__init__(path, split, feature_cache_name, **kwargs)
        if self.path is not None:
            self.graph_path = self.path / f'graph.bin'
            self.graph = dgl.load_graphs(str(self.graph_path))

    def load(self, path: str, split: str):
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
        self.nodes = []

        data_path = path / f'{split}.json'
        logger.info(f'loading data from {data_path}')
        data = json.load(open(data_path, 'r'))
        for i, item in tqdm(data.items()):
            self.ids.append(i)
            self.labels.append(item['label'])
            self.weak_labels.append(item['weak_labels'])
            self.examples.append(item['data'])
            self.nodes.append(item['data']['node_id'])
        node = self.nodes
        label_path = self.path / f'label.json'
        self.id2label = {int(k): v for k, v in json.load(open(label_path, 'r')).items()}

        return self
                

class GraphNumericDataset(GraphDataset, NumericDataset):
    """Data class for numeric dataset."""
    def __init__(self,
                 path: str = None,
                 split: Optional[str] = None,
                 feature_cache_name: Optional[str] = None,
                 **kwargs: Any) -> None:
        GraphDataset.__init__(self, path, split, feature_cache_name, **kwargs)


class GraphTextDataset(GraphDataset, TextDataset):
    """Data class for text graph node classification dataset."""

    def __init__(self,
                 path: str = None,
                 split: Optional[str] = None,
                 feature_cache_name: Optional[str] = None,
                 **kwargs: Any) -> None:
        GraphDataset.__init__(self, path, split, feature_cache_name,  **kwargs)