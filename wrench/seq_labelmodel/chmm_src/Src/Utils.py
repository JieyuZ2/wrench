import numpy as np
import torch


def log_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    a : m obs n
    b : n obs p

    output : m obs p matrix

    Normally, a matrix multiplication
    computes out_{i,j} = sum_k A_{i,k} obs B_{k,j}

    A log domain matrix multiplication
    computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}

    This is needed for numerical stability when A and B are probability matrices.
    """
    a1 = a.unsqueeze(-1)
    b1 = b.unsqueeze(-3)
    return (a1 + b1).logsumexp(-2)


def log_maxmul(a, b):
    a1 = a.unsqueeze(-1)
    b1 = b.unsqueeze(-3)
    return (a1 + b1).max(-2)


def validate_prob(x, dim=-1):
    if (x <= 0).any():
        prob = normalize(x, dim=dim)
    elif (x.sum(dim=dim) != 1).any():
        prob = x / x.sum(dim=dim, keepdim=True)
    else:
        prob = x
    return prob


def normalize(x, dim=-1, epsilon=1e-6):
    result = x - x.min(dim=dim, keepdim=True)[0] + epsilon
    result = result / result.sum(dim=dim, keepdim=True)
    return result


# noinspection PyTypeChecker
def logsumexp(x, dim=None, keepdim=False):
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim=dim, keepdim=True)
    x = torch.where(
        (xm == np.inf) | (xm == -np.inf),
        xm,
        xm + torch.logsumexp(x - xm, dim=dim, keepdim=True)
    )
    return x if keepdim else x.squeeze(dim)
