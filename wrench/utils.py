import random
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
            ndim = batch[k].ndim
            if ndim > 1:
                if ndim == 2:
                    batch[k] = v[:, :batch_max_seq]
                else:
                    batch[k] = v[:, :batch_max_seq, :]
        return batch

    return collate_fn_trunc_pad
