from typing import Optional, Any, Union
import numpy as np
import scipy.sparse as ssp
from scipy.special import digamma

from wrench.basemodel import BaseLabelModel
from wrench.dataset import BaseDataset
from ..utils import create_tuples


def ibcc(tuples, a_v=4, b_v=1, alpha=1):
    num_items, num_workers, num_classes = tuples.max(axis=0) + 1

    y_is_one_kij = []
    y_is_one_kji = []
    for k in range(num_classes):
        selected = (tuples[:, 2] == k)
        coo_ij = ssp.coo_matrix((np.ones(selected.sum()), tuples[selected, :2].T), shape=(num_items, num_workers),
                                dtype=np.bool)
        y_is_one_kij.append(coo_ij.tocsr())
        y_is_one_kji.append(coo_ij.T.tocsr())

    # initialization
    prior_kl = np.eye(num_classes) * (a_v - b_v) + b_v
    n_jkl = np.empty((num_workers, num_classes, num_classes))

    # MV initialize Z
    z_ik = np.zeros((num_items, num_classes))
    for l in range(num_classes):
        z_ik[:, [l]] += y_is_one_kij[l].sum(axis=-1) + 1e-8
    z_ik /= z_ik.sum(axis=-1, keepdims=True)
    last_z_ik = z_ik.copy()

    for iteration in range(500):
        # E step
        Eq_log_pi_k = digamma(z_ik.sum(axis=0) + alpha)  # - digamma(num_items + num_classes * alpha)

        for l in range(num_classes):
            n_jkl[:, :, l] = y_is_one_kji[l].dot(z_ik)

        Eq_log_v_jkl = digamma(n_jkl + prior_kl[None, :, :]) - \
                       digamma(n_jkl.sum(axis=-1) + prior_kl.sum(axis=-1))[:, :, None]

        # M step
        last_z_ik[:] = z_ik

        z_ik[:] = Eq_log_pi_k
        for l in range(num_classes):
            z_ik += y_is_one_kij[l].dot(Eq_log_v_jkl[:, :, l])
        z_ik -= z_ik.max(axis=-1, keepdims=True)
        z_ik = np.exp(z_ik)
        z_ik /= z_ik.sum(axis=-1, keepdims=True)

        if np.allclose(last_z_ik, z_ik, atol=1e-3):
            break
    return z_ik


class IBCC(BaseLabelModel):
    def __init__(self,
                 alpha: Optional[float] = 1,
                 a_v: Optional[float] = 4,
                 b_v: Optional[float] = 1,
                 **kwargs: Any):
        super().__init__()
        self.hyperparas = {
            'alpha': alpha,
            'a_v': a_v,
            'b_v': b_v,
            **kwargs
        }

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            verbose: Optional[bool] = False,
            *args: Any,
            **kwargs: Any):
        pass

    def predict_proba(self,
                      dataset: Union[BaseDataset, np.ndarray],
                      **kwargs: Any):
        tuples = create_tuples(dataset)

        return ibcc(tuples, **self.hyperparas)
