import random
from collections import Counter
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .backbone import BertRelationClassifier, BertTextClassifier
from .dataset import BERTTorchTextClassDataset, BERTTorchRelationClassDataset, BaseDataset, TextDataset, RelationDataset


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_bert_model_class(dataset: BaseDataset):
    if isinstance(dataset, TextDataset):
        return BertTextClassifier
    if isinstance(dataset, RelationDataset):
        return BertRelationClassifier
    raise NotImplementedError


def get_bert_torch_dataset_class(dataset: BaseDataset):
    if isinstance(dataset, TextDataset):
        return BERTTorchTextClassDataset
    if isinstance(dataset, RelationDataset):
        return BERTTorchRelationClassDataset
    raise NotImplementedError


def array_to_marginals(y, cardinality=None):
    class_counts = Counter(y)
    if cardinality is None:
        sorted_counts = np.array([v for k, v in sorted(class_counts.items())])
    else:
        sorted_counts = np.zeros(len(cardinality))
        for i, c in enumerate(cardinality):
            sorted_counts[i] = class_counts.get(c, 0)
    marginal = sorted_counts / sum(sorted_counts)
    return marginal


def calc_cmi_matrix(y, L):
    n, m = L.shape
    lf_cardinality = [sorted(np.unique(L[:, i])) for i in range(m)]

    n_class = len(np.unique(y))
    c_idx_l = [y == c for c in range(n_class)]
    c_cnt_l = [np.sum(c_idx) for c_idx in c_idx_l]
    class_marginal = [c_cnt / n for c_cnt in c_cnt_l]

    cond_probs = np.zeros((n_class, m, max(map(len, lf_cardinality))))
    for c, c_idx in enumerate(c_idx_l):
        for i in range(m):
            card_i = lf_cardinality[i]
            cond_probs[c, i][:len(card_i)] = array_to_marginals(L[:, i][c_idx], card_i)

    cmi_matrix = -np.ones((m, m)) * np.inf
    for i in range(m):
        L_i = L[:, i]
        card_i = lf_cardinality[i]
        for j in range(i + 1, m):
            L_j = L[:, j]
            card_j = lf_cardinality[j]

            cmi_ij = 0.0
            for c, (c_idx, n_c) in enumerate(zip(c_idx_l, c_cnt_l)):
                cmi = 0.0
                for ci_idx, ci in enumerate(card_i):
                    for cj_idx, cj in enumerate(card_j):
                        p = np.sum(np.logical_and(L_i[c_idx] == ci, L_j[c_idx] == cj)) / n_c
                        if p > 0:
                            cur = p * np.log(p / (cond_probs[c, i, ci_idx] * cond_probs[c, j, cj_idx]))
                            cmi += cur

                cmi_ij += class_marginal[c] * cmi
            cmi_matrix[i, j] = cmi_matrix[j, i] = cmi_ij

    return cmi_matrix


def cluster_based_accuracy_variance(Y, L, cluster_labels):
    correct = Y == L
    acc_l = []
    cluster_idx = np.unique(cluster_labels)
    for cluster in cluster_idx:
        cluster_correct = correct[cluster_labels == cluster]
        cluster_acc = np.sum(cluster_correct) / len(cluster_correct)
        acc_l.append(cluster_acc)
    return np.var(acc_l)


def cross_entropy_with_probs(
        input: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
) -> torch.Tensor:
    """Calculate cross-entropy loss when targets are probabilities (floats), not ints.

    PyTorch's F.cross_entropy() method requires integer labels; it does accept
    probabilistic labels. We can, however, simulate such functionality with a for loop,
    calculating the loss contributed by each class and accumulating the results.
    Libraries such as keras do not require this workaround, as methods like
    "categorical_crossentropy" accept float labels natively.

    Note that the method signature is intentionally very similar to F.cross_entropy()
    so that it can be used as a drop-in replacement when target labels are changed from
    from a 1D tensor of ints to a 2D tensor of probabilities.

    Parameters
    ----------
    input
        A [num_points, num_classes] tensor of logits
    target
        A [num_points, num_classes] tensor of probabilistic target labels
    weight
        An optional [num_classes] array of weights to multiply the loss by per class
    reduction
        One of "none", "mean", "sum", indicating whether to return one loss per data
        point, the mean loss, or the sum of losses

    Returns
    -------
    torch.Tensor
        The calculated loss

    Raises
    ------
    ValueError
        If an invalid reduction keyword is submitted
    """
    if input.shape[1] == 1:
        input = input.squeeze()
        if target.ndim == 2:
            target = target[:, 1]
        return F.binary_cross_entropy_with_logits(input, target, weight=weight, reduction=reduction)
    else:

        if target.ndim == 1:
            return F.cross_entropy(input, target.long(), weight=weight, reduction=reduction)

        num_points, num_classes = input.shape
        # Note that t.new_zeros, t.new_full put tensor on same device as t
        cum_losses = input.new_zeros(num_points)
        for y in range(num_classes):
            target_temp = input.new_full((num_points,), y, dtype=torch.long)
            y_loss = F.cross_entropy(input, target_temp, reduction="none")
            if weight is not None:
                y_loss = y_loss * weight[y]
            cum_losses += target[:, y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")


def construct_collate_fn_trunc_pad(mask: str):
    def collate_fn_trunc_pad(batch: Dict):
        batch = torch.utils.data._utils.collate.default_collate(batch)
        batch_mask = batch[mask]
        batch_max_seq = batch_mask.sum(dim=1).max()
        for k, v in batch.items():
            if k not in ['weak_labels', 'features']:
                ndim = batch[k].ndim
                if ndim > 1:
                    if ndim == 2:
                        batch[k] = v[:, :batch_max_seq]
                    else:
                        batch[k] = v[:, :batch_max_seq, :]
        return batch

    return collate_fn_trunc_pad
