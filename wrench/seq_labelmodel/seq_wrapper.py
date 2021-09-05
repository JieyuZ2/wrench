import logging
from copy import deepcopy
from typing import Any, List, Optional

import numpy as np

from ..basemodel import BaseSeqModel
from ..dataset import BaseSeqDataset

logger = logging.getLogger(__name__)


# following funcs assume O is indexed a 0, and B is odd, I is even!

def bio_id_to_io_id(a):
    return np.where(a > 0, np.where(a % 2 == 0, a / 2, (a + 1) / 2), a).astype(int)


def io_id_to_bio_id(a):
    bio_ids = []
    last_io = -1
    for i in a:
        if i == 0:
            bio_ids.append(0)
        else:
            if i == last_io:
                bio_ids.append(int(i * 2))  # to I
            else:
                bio_ids.append(int(i * 2 - 1))  # to B
        last_io = i
    return bio_ids


def check_weak_labels_seq(dataset: BaseSeqDataset, bio_to_io=True):
    assert dataset.weak_labels is not None, f'Input dataset has no weak labels!'
    L, Y, indexes = dataset.flatten()
    if bio_to_io:
        L = bio_id_to_io_id(L)
        Y = bio_id_to_io_id(Y)
    return L, Y, indexes


class SeqLabelModelWrapper(BaseSeqModel):
    def __init__(self, label_model_class, **kwargs: Any):
        super().__init__()
        self.hyperparas = {'label_model_class': label_model_class}
        self.hyperparas.update(**kwargs)
        self.label_model = None

    @staticmethod
    def transform_labeling_function(dataset: BaseSeqDataset):
        dataset = deepcopy(dataset)
        for i in range(len(dataset)):
            for j in range(len(dataset.weak_labels[i])):
                # for a token, if all LFs label "O", we do not change anything;
                # if some LFs label non-"O", we treat other LFs (which label "O") abstaining.
                if np.max(dataset.weak_labels[i][j]) > 0:
                    dataset.weak_labels[i][j] = [x if x > 0 else -1 for x in dataset.weak_labels[i][j]]
        return dataset

    def fit(self,
            dataset_train: BaseSeqDataset,
            dataset_valid: Optional[BaseSeqDataset] = None,
            y_valid: Optional[List[List]] = None,
            bio_to_io: Optional[bool] = True,
            **kwargs: Any):

        self.bio_to_io = bio_to_io
        dataset_train = self.transform_labeling_function(dataset_train)
        L, _, indexes = check_weak_labels_seq(dataset_train, bio_to_io)

        if dataset_valid is not None:
            dataset_valid = self.transform_labeling_function(dataset_valid)
            _, y_valid, _ = check_weak_labels_seq(dataset_valid, bio_to_io)
        else:
            y_valid = None

        n_class = dataset_train.n_class
        if bio_to_io:
            n_class = n_class // 2 + 1
        self.n_class = n_class

        hyperparas = deepcopy(self.hyperparas)
        label_model_class = hyperparas.pop('label_model_class')
        self.label_model = label_model_class(**hyperparas)
        self.label_model.fit(dataset_train=L, y_valid=y_valid, n_class=n_class, **kwargs)

    def predict(self, dataset: BaseSeqDataset, weight: Optional[np.ndarray] = None,
                **kwargs: Any):
        dataset = self.transform_labeling_function(dataset)
        L, _, indexes = check_weak_labels_seq(dataset, self.bio_to_io)

        y_pred = self.label_model.predict(L)
        preds = [list(y_pred[start:end]) for (start, end) in zip(indexes[:-1], indexes[1:])]
        if self.bio_to_io:
            preds = [io_id_to_bio_id(i) for i in preds]
        return preds
