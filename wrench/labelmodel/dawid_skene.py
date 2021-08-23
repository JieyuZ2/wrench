from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import warnings
import numpy as np

from .baselabelmodel import BaseLabelModel
from ..dataset import BaseDataset
from .utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1


class DawidSkene(BaseLabelModel):
    def __init__(self, n_epochs: Optional[int] = 10000, **kwargs: Any):
        super().__init__()
        self.hyperparas = {
            'n_epochs': n_epochs,
        }

    def fit(self,
            dataset_train:Union[BaseDataset, np.ndarray],
            y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            balance: Optional[np.ndarray] = None,
            tol: Optional[float] = 1e-5,
            verbose: Optional[bool] = False,
            **kwargs: Any):

        self._update_hyperparas(**kwargs)

        L = check_weak_labels(dataset_train)
        balance = balance or self._init_balance(L, dataset_valid, y_valid)
        cardinality = len(balance)
        self.cardinality = cardinality

        Y_p = self._initialize_Y_p(L)
        L_aug = self._initialize_L_aug(L)

        iter = 0
        converged = False
        max_iter = self.hyperparas['n_epochs']
        old_class_marginals = None
        old_error_rates = None
        while not converged:
            iter += 1

            # M-step
            (class_marginals, error_rates) = self._m_step(L_aug, Y_p)

            # E-setp
            Y_p = self._e_step(L_aug, class_marginals, error_rates)

            # # check likelihood
            # log_L = self._calc_likelihood(L_aug, class_marginals, error_rates)

            # check for convergence
            if old_class_marginals is not None:
                class_marginals_diff = np.sum(np.abs(class_marginals - old_class_marginals))
                error_rates_diff = np.sum(np.abs(error_rates - old_error_rates))
                if (class_marginals_diff < tol and error_rates_diff < tol) or iter > max_iter:
                    converged = True

            # update current values
            old_class_marginals = class_marginals
            old_error_rates = error_rates

        self.error_rates = error_rates
        self.class_marginals = class_marginals

    def _initialize_Y_p(self, L):
        cardinality = self.cardinality
        n, m = L.shape
        Y_p = np.zeros((n, cardinality))
        for i in range(n):
            counts = np.zeros(cardinality)
            for j in range(m):
                if L[i, j] != ABSTAIN:
                    counts[L[i, j]] += 1
            if counts.sum() == 0:
                counts += 1
            Y_p[i, :] = counts
        Y_p /= Y_p.sum(axis=1, keepdims=True)
        return Y_p

    def _initialize_L_aug(self, L):
        L_off = L + 1
        L_aug = (np.arange(self.cardinality+1) == L_off[...,None]).astype(int)
        return L_aug

    def _m_step(self, L_aug, Y_p):
        n, m, _ = L_aug.shape
        cardinality = self.cardinality
        class_marginals = np.sum(Y_p,0) / n

        # compute error rates
        error_rates = np.zeros([m, cardinality, cardinality+1])
        for k in range(m):
            for j in range(cardinality):
                for l in range(cardinality+1):
                    error_rates[k, j, l] = np.dot(Y_p[:, j], L_aug[:, k, l])
                # normalize by summing over all observation classes
                sum_over_responses = np.sum(error_rates[k, j, :])
                if sum_over_responses > 0:
                    error_rates[k, j, :] = error_rates[k, j, :] / float(sum_over_responses)

        return (class_marginals, error_rates)

    def _e_step(self, L_aug, class_marginals, error_rates):
        n, m, _ = L_aug.shape
        cardinality = self.cardinality

        Y_p = np.zeros([n, cardinality])

        for i in range(n):
            for j in range(cardinality):
                estimate = class_marginals[j]
                estimate *= np.prod(np.power(error_rates[:, j, :], L_aug[i, :, :]))

                Y_p[i, j] = estimate
            # normalize error rates by dividing by the sum over all observation classes
            Y_p_sum = np.sum(Y_p[i, :])
            if Y_p_sum > 0:
                Y_p[i, :] = Y_p[i, :] / Y_p_sum

        return Y_p

    def _calc_likelihood(self, L_aug, class_marginals, error_rates):
        n, m, _ = L_aug.shape
        cardinality = self.cardinality
        log_L = 0.0

        for i in range(n):
            single_likelihood = 0.0
            for j in range(cardinality):
                class_prior = class_marginals[j]
                Y_p_likelihood = np.prod(np.power(error_rates[:, j, :], L_aug[i, :, :]))
                Y_p_posterior = class_prior * Y_p_likelihood
                single_likelihood += Y_p_posterior

            temp = log_L + np.log(single_likelihood)

            if np.isnan(temp) or np.isinf(temp):
                warnings.warn('!')

            log_L = temp

        return log_L

    def predict_proba(self, dataset:Union[BaseDataset, np.ndarray], **kwargs: Any) -> np.ndarray:
        L = check_weak_labels(dataset)
        L_aug = self._initialize_L_aug(L)
        Y_p = self._e_step(L_aug, self.class_marginals, self.error_rates)
        return Y_p