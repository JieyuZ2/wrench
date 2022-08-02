import logging
from copy import deepcopy
from typing import List

import numpy as np

from wrench.dataset import load_dataset, BaseDataset
from wrench.labelmodel import Weapo
from wrench._logging import LoggingHandler

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)


def get_pu_dataset(datasets: List[BaseDataset], positive_label=1, drop_lf_idx=[]):
    processed_datasets = []
    L = np.array(datasets[0].weak_labels)
    y = np.array(datasets[0].labels)
    source_idx = np.nonzero(np.any(L == positive_label, axis=0))[0]

    if np.any(y):
        prior = np.mean(y == positive_label)
        source_idx_ = []
        for i in source_idx:
            if np.mean(y[L[:, i] == positive_label] == positive_label) >= prior:
                source_idx_.append(i)
        source_idx = np.array(source_idx_)
    source_idx = [i for i in source_idx if i not in drop_lf_idx]

    for dataset in datasets:
        dataset = deepcopy(dataset)
        weak_labels = np.array(dataset.weak_labels)
        weak_labels = weak_labels[:, source_idx]
        pos_idx = weak_labels == positive_label
        weak_labels[pos_idx] = 1
        weak_labels[~pos_idx] = 0
        dataset.weak_labels = weak_labels
        dataset.n_lf = len(source_idx)
        processed_datasets.append(dataset)
    return processed_datasets


#### Load dataset
dataset_path = '../datasets/'
data = 'mushroom'
train_data, valid_data, test_data = get_pu_dataset(load_dataset(
    dataset_path,
    data,
    extract_feature=False,
))

#### Run label model: Snorkel
label_model = Weapo(
    prior_cons=True,
)
label_model.fit(
    dataset_train=train_data,
    dataset_valid=valid_data
)
auc = label_model.test(test_data, 'auc')
logger.info(f'label model test auc: {auc}')

