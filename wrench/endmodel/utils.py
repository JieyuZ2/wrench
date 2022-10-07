import numpy as np
from typing import List

from wrench.basemodel import BaseTorchModel, BaseClassModel
from wrench.dataset import BaseDataset
from sklearn.metrics import f1_score
from snorkel.utils import probs_to_preds


def calc_prior(labels: List, n_class: int):
    return [labels.count(i) for i in range(n_class)]


def create_unbalanced_set(data: BaseDataset, imbalance_ratio: int):
    miu = (1 / imbalance_ratio) ** (1 / (data.n_class-1))
    ids = np.argsort(data.labels)
    prior = np.array(calc_prior(data.labels, data.n_class))
    imbalance_list = np.array([int(n * miu ** i) for (i, n) in enumerate(prior)])  # n_i * μ^i
    print(imbalance_list)
    prior_cumsum = np.cumsum(prior)
    prior_cumsum = np.insert(prior_cumsum, 0, 0)

    sampled_ids = np.concatenate([np.random.choice(ids[prior_cumsum[i]:prior_cumsum[i + 1]], n)
                                  for i, n in enumerate(imbalance_list)])

    return sampled_ids


def calc_f1(data: BaseDataset, model: BaseClassModel):
    probas = model.predict_proba(data)
    y_pred = probs_to_preds(probas)
    y_true = np.array(data.labels)
    return f1_score(y_true, y_pred, average=None)
