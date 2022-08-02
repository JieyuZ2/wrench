import logging
from itertools import combinations
from typing import Any, Optional, Union

import cvxpy as cp
import networkx as nx
import numpy as np
from tqdm import tqdm

from ..basemodel import BaseLabelModel
from ..dataset import BaseDataset
from ..dataset.utils import check_weak_labels

logger = logging.getLogger(__name__)

ABSTAIN = -1


def covered_by_(l1, l2):
    left_cover = False
    right_cover = False
    for l1i, l2i in zip(l1, l2):
        if l1i > l2i:
            left_cover = True
        if l2i > l1i:
            right_cover = True
        if left_cover and right_cover:
            return False
    return right_cover


def get_partial_order_tree(unique_L, exclude_all_abstain=True):
    tree = nx.DiGraph()
    if exclude_all_abstain:
        for i, li in enumerate(unique_L):
            for j, lj in enumerate(unique_L):
                if i != j and sum(li) > 0 and covered_by_(li, lj):
                    tree.add_edge(j, i)
    else:
        for i, li in enumerate(unique_L):
            for j, lj in enumerate(unique_L):
                if i != j and covered_by_(li, lj):
                    tree.add_edge(j, i)
    return tree


def get_binary_constraints(L, min_cnt=0.0, exclude_all_abstain=True):
    min_cnt = min_cnt * np.sum(L.sum(1) > 0)

    unique_L, inv = np.unique(L, axis=0, return_inverse=True)
    unique_L = [tuple(i) for i in unique_L]
    data_idx = [tuple(np.where(inv == i)[0]) for i in range(len(unique_L))]

    g = set()
    partial_order_tree = get_partial_order_tree(unique_L, exclude_all_abstain)
    if len(partial_order_tree.edges) == 0:
        return g

    node2descendants = {n: list(nx.descendants(partial_order_tree, n)) for n in partial_order_tree.nodes}
    for u in partial_order_tree.nodes:
        successors = list(partial_order_tree.successors(u))
        descendants = node2descendants[u]
        for v in successors:
            for d in descendants:
                if v in node2descendants[d]:
                    partial_order_tree.remove_edge(u, v)
                    break

    for u, v in partial_order_tree.edges:
        idx1, idx2 = data_idx[u], data_idx[v]
        if len(idx1) > min_cnt and len(idx2) > min_cnt:
            g.add((idx1, idx2))

    return g


def process_fn(i, L_i, exclude_all_abstain=True):
    idx_l = list(range(L_i.shape[1]))
    G = set()
    for l_idx in combinations(idx_l, i):
        g = get_binary_constraints(L_i[:, l_idx], exclude_all_abstain=exclude_all_abstain)
        G.update(g)
    return G


def get_components(L, single=False):
    # extract min isolated sets
    A = L.T @ L
    m = L.shape[1]
    g = nx.Graph()
    for i in range(m):
        for j in range(i, m):
            if A[i, j] > 0:
                g.add_edge(i, j)

    # filer sets with size = 1
    components = []
    for c in nx.algorithms.connected_components(g):
        if len(c) > 1:
            L_c = L[:, list(c)]
            u = [i for i in np.unique(L_c, axis=0) if sum(i) > 0]
            assert len(u) > 1
            # remove dub has bug if len(u)==1
            components.append(list(c))

    if single:
        single_lf_idx = []
        for i in range(m):
            if np.sum(A[i]) == A[i, i]:
                single_lf_idx.append([i])
        components.extend(single_lf_idx)

    return components


def get_constraints(L, full=False):
    # remove dup
    L_uni, idx_uni = np.unique(L, axis=1, return_index=True)
    L = L_uni

    components = get_components(L)

    from multiprocessing import Pool
    from functools import partial

    if full:
        G_s = get_binary_constraints(L, exclude_all_abstain=True)
    else:
        G_s = set()

        for c in components:
            L_c = L[:, list(c)]
            range_m = list(range(2, L_c.shape[1] + 1))
            if len(range_m) >= 5:
                pool = Pool(len(range_m))
                worker = partial(process_fn, L_i=L_c)
                for val in tqdm(pool.imap_unordered(worker, range_m), total=len(range_m)):
                    g = val
                    G_s.update(g)
                # pool.join()
                # pool.close()
            else:
                for i in tqdm(range_m):
                    g = process_fn(i, L_i=L_c)
                    G_s.update(g)

    G_b = np.zeros((len(G_s), len(L)))
    for i, (d1, d2) in enumerate(G_s):
        G_b[i, d1] = -1 / len(d1)
        G_b[i, d2] = 1 / len(d2)

    return G_b


class Weapo(BaseLabelModel):
    def __init__(self, prior_cons=True, **kwargs: Any):
        super().__init__()
        self.prior_cons = prior_cons
        self.w = None

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Optional[Union[BaseDataset, np.ndarray]] = None,
            y_valid: Optional[np.ndarray] = None,
            balance: Optional[np.ndarray] = None,
            verbose: Optional[bool] = True,
            **kwargs: Any):
        L = check_weak_labels(dataset_train)
        if balance is None:
            balance = self._init_balance(L, dataset_valid, y_valid, 2)
        self.balance = balance

        n, m = L.shape
        _, a, b, c = np.unique(L, True, True, True, axis=1)
        B = []
        equal_indices = []
        for i, cnt in enumerate(c):
            if cnt > 1:
                indices = np.where(b == i)[0]
                equal_indices.append(indices)
                for (v1, v2) in combinations(indices, 2):
                    v = np.zeros(m)
                    v[v1] = 1
                    v[v2] = -1
                    B.append(v)

        components = get_components(L, single=True)
        A = np.zeros((len(components), m))
        for i, c in enumerate(components):
            A[i, c] = 1

        G = get_constraints(L, full=True)

        w = cp.Variable((m, 1), nonneg=True)
        P = G @ L
        psi = P @ w

        G_pos_cnt = np.sum(G > 0, 1)
        G_neg_cnt = np.sum(G < 0, 1)
        G_cnt = np.minimum(G_pos_cnt, G_neg_cnt)

        G_cnt = G_cnt / G_cnt.sum()
        margin = cp.sum(cp.pos(cp.multiply(G_cnt, psi.flatten())))

        x = L[L.sum(1) > 0] @ w
        x_mean = cp.sum(x) / x.shape[0]
        cover_rate = np.mean(L.sum(1) > 0)
        lower = balance[1]
        upper = min(1, balance[1] / cover_rate)

        if self.prior_cons:
            obj = cp.Minimize(margin + cp.norm(w, 2)
                              + cp.pos(lower - x_mean)
                              + cp.pos(x_mean - upper)
                              )
        else:
            obj = cp.Minimize(margin)

        constraints = [A @ w == 1]

        if len(B) > 0:
            B = np.array(B)
            constraints.append(B @ w == 0)

        prob = cp.Problem(obj, constraints)

        # prob.solve(cp.ECOS, max_iters=1000)
        prob.solve(cp.SCS)
        assert w.value is not None
        self.w = w.value

        self.L = L
        self.G = G
        self.P = P
        self.A = A
        self.B = B
        self.components = components

    def predict_proba(self, dataset: Union[BaseDataset, np.ndarray], **kwargs: Any) -> np.ndarray:
        L = check_weak_labels(dataset)
        proba = np.zeros((len(L), 2))
        raw_score = (L @ self.w)
        max_ = max(raw_score)
        min_ = min(raw_score)

        proba[:, 1] = ((raw_score - min_) / (max_ - min_)).flatten()

        proba[:, 0] = 1 - proba[:, 1]
        return proba
