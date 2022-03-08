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
        super().load(self.path, self.split)
        self.node_id = []
        data_path = path / f'{split}.json'
        data = json.load(open(data_path, 'r'))
        for i, item in tqdm(data.items()):
            self.node_id.append(item["data"]["node_id"])
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