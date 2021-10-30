import logging
from typing import Any, List, Optional, Union

import numpy as np
import torch
from metal.label_model import LabelModel as LabelModel_

from ..basemodel import BaseLabelModel
from ..dataset import BaseDataset
from ..dataset.utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1


class LabelModel(LabelModel_):
    def _build_mask(self):
        """Build mask applied to O^{-1}, O for the matrix approx constraint"""
        self.mask = torch.ones(self.d, self.d).bool()
        for ci in self.c_data.values():
            si, ei = ci["start_index"], ci["end_index"]
            for cj in self.c_data.values():
                sj, ej = cj["start_index"], cj["end_index"]

                # Check if ci and cj are part of the same maximal clique
                # If so, mask out their corresponding blocks in O^{-1}
                if len(ci["max_cliques"].intersection(cj["max_cliques"])) > 0:
                    self.mask[si:ei, sj:ej] = 0
                    self.mask[sj:ej, si:ei] = 0

    def _set_dependencies(self, deps):
        super()._set_dependencies(deps)
        if len(deps) > 0:
            self.higher_order = True
        else:
            self.higher_order = False

    def _generate_O(self, L):
        """Form the overlaps matrix, which is just all the different observed
        combinations of values of pairs of sources
        Note that we only include the k non-abstain values of each source,
        otherwise the model not minimal --> leads to singular matrix
        """
        L_aug = self._get_augmented_label_matrix(L, higher_order=self.higher_order)
        self.d = L_aug.shape[1]
        self.O = torch.from_numpy(L_aug.T @ L_aug / self.n).float()

    def predict_proba(self, L):
        """Returns the [n,k] matrix of label probabilities P(Y | \lambda)
        Args:
            L: An [n,m] scipy.sparse label matrix with values in {0,1,...,k}
        """
        self._set_constants(L)

        L_aug = self._get_augmented_label_matrix(L, higher_order=self.higher_order)
        mu = np.clip(self.mu.detach().clone().numpy(), 0.01, 0.99)

        # Create a "junction tree mask" over the columns of L_aug / mu
        if len(self.deps) > 0:
            L_aug = self._get_augmented_label_matrix(L, higher_order=self.higher_order)
            mu = np.clip(self.mu.detach().clone().numpy(), 0.01, 0.99)
            jtm = np.zeros(L_aug.shape[1])

            # All maximal cliques are +1
            for i in self.c_tree.nodes():
                node = self.c_tree.node[i]
                jtm[node["start_index"]: node["end_index"]] = 1

            # All separator sets are -1
            for i, j in self.c_tree.edges():
                edge = self.c_tree[i][j]
                jtm[edge["start_index"]: edge["end_index"]] = 1
        else:
            jtm = np.ones(L_aug.shape[1])

        # Note: We omit abstains, effectively assuming uniform distribution here
        X = np.exp(L_aug @ np.diag(jtm) @ np.log(mu) + np.log(self.p))
        Z = np.tile(X.sum(axis=1).reshape(-1, 1), self.k)
        return X / Z


class MeTaL(BaseLabelModel):
    def __init__(self,
                 lr: Optional[float] = 0.001,
                 l2: Optional[float] = 0.0,
                 n_epochs: Optional[int] = 1000,
                 seed: Optional[int] = None,
                 **kwargs: Any):
        super().__init__()
        self.hyperparas = {
            'lr'      : lr,
            'l2'      : l2,
            'n_epochs': n_epochs,
            'seed'    : seed or np.random.randint(1e6),
        }
        self.model = None

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            balance: Optional[np.ndarray] = None,
            dependency_graph: Optional[List] = [],
            verbose: Optional[bool] = False,
            **kwargs: Any):

        self._update_hyperparas(**kwargs)
        if isinstance(dataset_train, BaseDataset):
            if n_class is not None:
                assert n_class == dataset_train.n_class
            else:
                n_class = dataset_train.n_class
        if n_class is not None and balance is not None:
            assert len(balance) == n_class

        L = check_weak_labels(dataset_train)
        if balance is None:
            balance = self._init_balance(L, dataset_valid, y_valid, n_class)
        n_class = len(balance)
        self.n_class = n_class

        label_model = LabelModel(k=n_class, seed=self.hyperparas['seed'])
        label_model.train_model(
            L_train=L + 1,
            class_balance=balance,
            deps=dependency_graph,
            n_epochs=self.hyperparas['n_epochs'],
            lr=self.hyperparas['lr'],
            l2=self.hyperparas['l2'],
            seed=self.hyperparas['seed'],
            verbose=verbose
        )

        self.model = label_model

    def predict_proba(self, dataset: Union[BaseDataset, np.ndarray], **kwargs: Any) -> np.ndarray:
        L = check_weak_labels(dataset)
        return self.model.predict_proba(L + 1)
