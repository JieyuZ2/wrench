import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def consistency_loss(logits_w1, logits_w2):
    logits_w2 = logits_w2.detach()
    assert logits_w1.size() == logits_w2.size()
    return F.mse_loss(torch.softmax(logits_w1, dim=-1), torch.softmax(logits_w2, dim=-1), reduction='mean')


class BatchNormController:
    def __init__(self):
        """
        freeze_bn and unfreeze_bn must appear in pairs
        """
        self.backup = {}

    def freeze_bn(self, model):
        assert self.backup == {}
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                self.backup[name + '.running_mean'] = m.running_mean.data.clone()
                self.backup[name + '.running_var'] = m.running_var.data.clone()
                self.backup[name + '.num_batches_tracked'] = m.num_batches_tracked.data.clone()

    def unfreeze_bn(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.running_mean.data = self.backup[name + '.running_mean']
                m.running_var.data = self.backup[name + '.running_var']
                m.num_batches_tracked.data = self.backup[name + '.num_batches_tracked']
        self.backup = {}


def exp_rampup(current, rampup_epochs):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_epochs == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_epochs)
        phase = 1.0 - current / rampup_epochs
        return float(np.exp(-5.0 * phase * phase))
