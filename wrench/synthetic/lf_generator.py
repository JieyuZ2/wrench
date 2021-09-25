from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from itertools import chain
from typing import List, Optional, Union, Callable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import check_random_state

from ..dataset import BaseDataset, TextDataset
from ..utils import array_to_marginals, calc_cmi_matrix, cluster_based_accuracy_variance

ABSTAIN = -1


class Expression(ABC):
    @abstractmethod
    def apply(self, x: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def include(self, other):
        raise NotImplementedError

    @abstractmethod
    def exclude(self, other):
        raise NotImplementedError

    def overlap(self, other):
        if self.exclude(other): return False
        if self.include(other): return False
        if other.include(self): return False
        return True


class UnaryExpression(Expression):
    def __init__(self, idx, threshold):
        self.idx = idx
        self.threshold = threshold

    def apply(self, x: np.ndarray):
        assert x.ndim == 2, 'dimension of x should be 2!'
        return self.apply_(x[:, self.idx])

    @abstractmethod
    def apply_(self, x: np.ndarray):
        raise NotImplementedError

    def include(self, other: Expression):
        if isinstance(other, UnaryExpression):
            if self.idx == other.idx:
                return self.include_(other)
            return False
        if isinstance(other, BinaryExpression):
            return self.include(other.e1) and self.include(other.e2)

    @abstractmethod
    def include_(self, other: Expression):
        raise NotImplementedError

    def exclude(self, other: Expression):
        if isinstance(other, UnaryExpression):
            if self.idx == other.idx:
                return self.exclude_(other)
            return True
        if isinstance(other, BinaryExpression):
            return self.exclude(other.e1) and self.exclude(other.e2)

    @abstractmethod
    def exclude_(self, other: Expression):
        raise NotImplementedError

    def __str__(self):
        s = f'=====[{self.__class__}]=====\n'
        s += f'[idx] {self.idx}\n'
        s += f'[threshold] {self.threshold}\n'
        return s


class GreaterExpression(UnaryExpression):
    def apply_(self, x: np.ndarray):
        return x > self.threshold

    def include_(self, other: Expression):
        if isinstance(other, GreaterExpression) or isinstance(other, EqualExpression):
            return other.threshold > self.threshold
        if isinstance(other, InIntervalExpression):
            return other.threshold[0] > self.threshold
        return False

    def exclude_(self, other: Expression):
        if isinstance(other, EqualExpression):
            return other.threshold < self.threshold
        if isinstance(other, InIntervalExpression):
            return other.threshold[1] < self.threshold
        if isinstance(other, LessExpression):
            return other.threshold < self.threshold
        return False


class LessExpression(UnaryExpression):
    def apply_(self, x: np.ndarray):
        return x < self.threshold

    def include_(self, other: Expression):
        if isinstance(other, LessExpression) or isinstance(other, EqualExpression):
            return other.threshold < self.threshold
        if isinstance(other, InIntervalExpression):
            return other.threshold[1] < self.threshold
        return False

    def exclude_(self, other: Expression):
        if isinstance(other, EqualExpression):
            return other.threshold > self.threshold
        if isinstance(other, InIntervalExpression):
            return other.threshold[0] > self.threshold
        if isinstance(other, GreaterExpression):
            return other.threshold > self.threshold
        return False


class EqualExpression(UnaryExpression):
    def apply_(self, x: np.ndarray):
        return x == self.threshold

    def include_(self, other: Expression):
        if isinstance(other, EqualExpression):
            return other.threshold == self.threshold
        return False

    def exclude_(self, other: Expression):
        if isinstance(other, EqualExpression):
            return other.threshold != self.threshold
        else:
            return other.exclude(self)


class InIntervalExpression(UnaryExpression):
    def apply_(self, x: np.ndarray):
        return np.logical_and(self.threshold[0] < x, x < self.threshold[1])

    def include_(self, other: Expression):
        if isinstance(other, EqualExpression):
            return self.threshold[0] < other.threshold < self.threshold[1]
        if isinstance(other, InIntervalExpression):
            return self.threshold[0] < other.threshold[0] and other.threshold[1] < self.threshold[1]
        return False

    def exclude_(self, other: Expression):
        if isinstance(other, EqualExpression):
            return other.threshold < self.threshold[0] or other.threshold > self.threshold[1]
        if isinstance(other, InIntervalExpression):
            return other.threshold[0] > self.threshold[1] or other.threshold[1] < self.threshold[0]
        return other.exclude(self)


class OutIntervalExpression(UnaryExpression):
    def apply_(self, x: np.ndarray):
        return np.logical_or(self.threshold[0] > x, x > self.threshold[1])

    def include_(self, other: Expression):
        if isinstance(other, EqualExpression):
            return self.threshold[0] > other.threshold or other.threshold > self.threshold[1]
        if isinstance(other, GreaterExpression):
            return self.threshold[1] < other.threshold
        if isinstance(other, LessExpression):
            return self.threshold[0] > other.threshold
        if isinstance(other, InIntervalExpression):
            return self.threshold[0] > other.threshold[1] or other.threshold[0] > self.threshold[1]
        if isinstance(other, OutIntervalExpression):
            return self.threshold[0] > other.threshold[0] and other.threshold[1] > self.threshold[1]
        return False

    def exclude_(self, other: Expression):
        if isinstance(other, EqualExpression):
            return self.threshold[0] < other.threshold < self.threshold[1]
        if isinstance(other, InIntervalExpression):
            return other.threshold[0] > self.threshold[0] and other.threshold[1] < self.threshold[1]
        return False


class BinaryExpression(Expression):
    logic_op: Callable

    def __init__(self, e1: Expression, e2: Expression):
        self.e1 = e1
        self.e2 = e2

    def apply(self, x: np.ndarray):
        x1 = self.e1.apply(x)
        x2 = self.e2.apply(x)
        return self.logic_op(x1, x2)

    def include(self, other: Expression):
        if isinstance(other, UnaryExpression):
            return self.e1.include(other) or self.e2.include(other)
        if isinstance(other, BinaryExpression):
            e1_included = self.e1.include(other.e1) or self.e2.include(other.e1)
            e2_included = self.e1.include(other.e2) or self.e2.include(other.e2)
            return e1_included and e2_included

    def exclude(self, other: Expression):
        if isinstance(other, UnaryExpression):
            return self.e1.exclude(other) and self.e2.exclude(other)
        if isinstance(other, BinaryExpression):
            e1_excluded = self.e1.exclude(other.e1) and self.e2.exclude(other.e1)
            e2_excluded = self.e1.exclude(other.e2) and self.e2.exclude(other.e2)
            return e1_excluded and e2_excluded


class AndExpression(BinaryExpression):
    logic_op = staticmethod(np.logical_and)


class OrExpression(BinaryExpression):
    logic_op = staticmethod(np.logical_or)


class NGramExpression(Expression):
    def __init__(self, idx, threshold, ngram):
        self.idx = idx
        self.threshold = threshold
        self.ngram = ngram

    def apply(self, x: np.ndarray):
        assert x.ndim == 2, 'dimension of x should be 2!'
        applied = x[:, self.idx] > self.threshold
        if isinstance(applied, csr_matrix):
            applied = applied.toarray().squeeze()
        return applied

    def include(self, other):
        raise NotImplementedError

    def exclude(self, other):
        raise NotImplementedError

    def __str__(self):
        s = f'=====[{self.__class__}]=====\n'
        s += f'[idx] {self.idx}\n'
        s += f'[threshold] {self.threshold}\n'
        s += f'[ngram] {self.ngram}\n'
        return s


class LF:
    def __init__(self, e: Expression, label: int, acc: float = -1.0, propensity: float = -1.0):
        self.e = e
        self.label = label
        self.acc = acc
        self.propensity = propensity

    def apply(self, x: np.ndarray):
        x = self.e.apply(x)
        return x * self.label + (1 - x) * ABSTAIN


class AbstractLFApplier:
    def __init__(self, lf_list: List[LF]):
        self.lfs = lf_list
        self.labels = [r.label for r in lf_list]
        self.accs = [r.acc for r in lf_list]

    @abstractmethod
    def apply(self, dataset):
        raise NotImplementedError

    def __len__(self):
        return len(self.lfs)


class FeatureLFApplier(AbstractLFApplier):
    def __init__(self, lf_list: List[LF], preprocessor: Optional[Callable] = None):
        super().__init__(lf_list)
        self.preprocessor = preprocessor

    def apply(self, dataset: Union[BaseDataset, np.ndarray]):
        if self.preprocessor is not None:
            X = self.preprocessor(dataset)
        else:
            if isinstance(dataset, BaseDataset):
                X = np.array(dataset.features)
            else:
                X = dataset
        L = np.stack([lf.apply(X) for lf in self.lfs]).T
        return L


class NGramLFApplier(AbstractLFApplier):
    def __init__(self, lf_list: List[LF], vectorizer: CountVectorizer):
        super().__init__(lf_list)
        self.vectorizer = vectorizer

    def apply(self, dataset: Union[TextDataset, csr_matrix]):
        if isinstance(dataset, TextDataset):
            corpus = [i['text'] for i in dataset.examples]
            X = self.vectorizer.transform(corpus)
        else:
            X = dataset
        L = np.stack([lf.apply(X) for lf in self.lfs]).T
        return L


class NoEnoughLFError(Exception):
    def __init__(self, label=None):
        if label is None:
            self.message = 'cannot find enough lfs, please lower the min support or the min acc gain!'
        else:
            self.message = f'cannot find any lf for label {label}, please lower the min support or the min acc gain!'
        super().__init__(self.message)


class AbstractLFGenerator(ABC):
    lf_applier_type: Callable
    X: Union[np.ndarray, csr_matrix]
    label_to_candidate_lfs: dict

    def __init__(self,
                 dataset: Union[BaseDataset, np.ndarray],
                 y: Optional[np.ndarray] = None,
                 min_acc_gain: float = 0.1,
                 min_support: float = 0.01,
                 random_state=None
                 ):
        if isinstance(dataset, BaseDataset):
            self.Y = np.array(dataset.labels)
        else:
            assert y is not None
            self.Y = y
        self.n_class = len(set(self.Y))
        assert self.n_class > 1
        self.dataset = dataset
        self.n_data = len(dataset)
        self.min_support = int(min_support * self.n_data)
        self.min_acc_gain = min_acc_gain
        self.class_marginal = array_to_marginals(self.Y)

        self.generator = check_random_state(random_state)

    @staticmethod
    def calc_acc(y):
        return np.sum(y) / len(y)

    def check_candidate_lfs_enough_(self, n_lfs: Union[int, List[int]]):
        if isinstance(n_lfs, int):
            assert sum(map(len, self.label_to_candidate_lfs.values())) > n_lfs, NoEnoughLFError()
        else:
            assert len(n_lfs) == self.n_class
            labels = list(range(self.n_class))
            for label, n_lfs_i in zip(labels, n_lfs):
                assert len(self.label_to_candidate_lfs[label]) > n_lfs_i, NoEnoughLFError(label)

    def return_candidate_lfs(self):
        return list(chain.from_iterable(self.label_to_candidate_lfs.values()))

    def generate(self, mode: str, **kwargs):
        if mode == 'exhaustive':
            return self.exhaustive_generate()
        if mode == 'random':
            return self.random_generate(**kwargs)
        if mode == 'accurate':
            return self.accurate_generate(**kwargs)
        if mode == 'correlated':
            return self.correlated_generate(**kwargs)
        if mode == 'cluster_dependent':
            return self.cluster_dependent_generate(**kwargs)
        raise NotImplementedError(f'generate mode {mode} is not implemented!')

    def exhaustive_generate(self) -> AbstractLFApplier:
        return self.lf_applier_type(self.return_candidate_lfs())

    def random_generate(self, n_lfs: Union[int, List[int]] = 10, duplicated_lf=False) -> AbstractLFApplier:
        if not duplicated_lf:
            self.check_candidate_lfs_enough_(n_lfs)
        if isinstance(n_lfs, int):
            candidate_lfs = self.return_candidate_lfs()
            lfs = list(self.generator.choice(candidate_lfs, n_lfs, replace=duplicated_lf))
        else:
            labels = list(range(self.n_class))
            lfs = []
            for label, n_lfs_i in zip(labels, n_lfs):
                candidate_lfs = self.label_to_candidate_lfs[label]
                lfs_i = list(self.generator.choice(candidate_lfs, n_lfs_i, replace=duplicated_lf))
                lfs += lfs_i
        return self.lf_applier_type(lfs)

    def accurate_generate(self, n_lfs: Union[int, List[int]] = 10) -> AbstractLFApplier:
        self.check_candidate_lfs_enough_(n_lfs)
        if isinstance(n_lfs, int):
            candidate_lfs = self.return_candidate_lfs()
            lfs = sorted(candidate_lfs, key=lambda x: -x.acc)[:n_lfs]
        else:
            labels = list(range(self.n_class))
            lfs = []
            for label, n_lfs_i in zip(labels, n_lfs):
                candidate_lfs = self.label_to_candidate_lfs[label]
                lfs += sorted(candidate_lfs, key=lambda x: -x.acc)[:n_lfs_i]
        return self.lf_applier_type(lfs)

    def correlated_generate(self,
                            n_lfs: Union[int, List[int]] = 20,
                            # n_correlated_lfs: Union[int, List[int]] = 10,
                            ) -> AbstractLFApplier:
        # assert type(n_lfs) == type(n_correlated_lfs)
        self.check_candidate_lfs_enough_(n_lfs)
        if isinstance(n_lfs, int):
            candidate_lfs = self.return_candidate_lfs()
            L = np.stack([lf.apply(self.X) for lf in candidate_lfs]).T

            cmi_matrix = calc_cmi_matrix(self.Y, L)

            row_max, col_max = np.unravel_index(cmi_matrix.argmax(), cmi_matrix.shape)
            lfs_idx = [row_max, col_max]
            while len(lfs_idx) < n_lfs:
                sub_cmi_matrix = cmi_matrix[lfs_idx, :]
                next_to_add = sub_cmi_matrix.mean(0).argmax()
                lfs_idx.append(next_to_add)

            lfs = [candidate_lfs[i] for i in lfs_idx]

        else:
            labels = list(range(self.n_class))
            lfs = []
            for label, n_lfs_i in zip(labels, n_lfs):
                candidate_lfs = self.label_to_candidate_lfs[label]
                L = np.stack([lf.apply(self.X) for lf in candidate_lfs]).T
                Y = np.array(self.Y == label, dtype=int)

                cmi_matrix = calc_cmi_matrix(Y, L)

                row_max, col_max = np.unravel_index(cmi_matrix.argmax(), cmi_matrix.shape)
                lfs_idx = [row_max, col_max]
                while len(lfs_idx) < n_lfs_i:
                    sub_cmi_matrix = cmi_matrix[lfs_idx, :]
                    next_to_add = sub_cmi_matrix.mean(0).argmax()
                    lfs_idx.append(next_to_add)

                lfs += [candidate_lfs[i] for i in lfs_idx]
        return self.lf_applier_type(lfs)

    def cluster_dependent_generate(self, n_lfs: Union[int, List[int]] = 10, n_clusters=5) -> AbstractLFApplier:
        self.check_candidate_lfs_enough_(n_lfs)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.generator).fit(self.X)
        cluster_labels = kmeans.labels_
        if isinstance(n_lfs, int):
            candidate_lfs = self.return_candidate_lfs()
            L = np.stack([lf.apply(self.X) for lf in candidate_lfs]).T
            acc_var = np.array(
                [cluster_based_accuracy_variance(self.Y, L[:, i], cluster_labels) for i in range(L.shape[1])])
            argsort_idx = np.argsort(-acc_var)
            lfs = [candidate_lfs[i] for i in argsort_idx[:n_lfs]]
        else:
            labels = list(range(self.n_class))
            lfs = []
            for label, n_lfs_i in zip(labels, n_lfs):
                candidate_lfs = self.label_to_candidate_lfs[label]
                L = np.stack([lf.apply(self.X) for lf in candidate_lfs]).T
                acc_var = np.array(
                    [cluster_based_accuracy_variance(self.Y, L[:, i], cluster_labels) for i in range(L.shape[1])])
                argsort_idx = np.argsort(-acc_var)
                lfs += [candidate_lfs[i] for i in argsort_idx[:n_lfs_i]]
        return self.lf_applier_type(lfs)


class FeatureLFGenerator(AbstractLFGenerator):
    def __init__(self,
                 dataset: Union[BaseDataset, np.ndarray],
                 y: Optional[np.ndarray] = None,
                 min_acc_gain: float = 0.1,
                 min_support: float = 0.01,
                 random_state=None
                 ):
        super(FeatureLFGenerator, self).__init__(dataset, y, min_acc_gain, min_support, random_state)
        if isinstance(dataset, BaseDataset):
            self.X = np.array(dataset.features)
        else:
            assert y is not None
            self.X = dataset
        self.n_feature = self.X.shape[1]
        self.bin_list = self.get_bin_egdes(self.X, self.min_support)
        self.label_to_candidate_lfs, self.idx_to_lfs, self.label_to_idx_to_lfs = self.generate_label_to_lfs()
        self.lf_applier_type = FeatureLFApplier

    @staticmethod
    def get_bin_egdes(X: np.ndarray, bin_size: int):
        n_data, n_features = X.shape
        bin_list = []
        for i in range(n_features):
            x = X[:, i]
            argsort_idx = np.argsort(x)
            min_x, max_x = np.min(x), np.max(x)
            bin_list_i = [min_x]
            interval = bin_size
            while interval < n_data:
                thres = x[argsort_idx[interval]]
                if thres == max_x: break
                while interval < n_data:
                    if x[argsort_idx[interval - 1]] == thres:
                        interval += 1
                        thres = x[argsort_idx[interval]]
                    else:
                        break
                if thres == max_x:
                    bin_list_i.append(max_x)
                    break
                else:
                    bin_list_i.append((thres + x[argsort_idx[interval + 1]]) / 2)
                    interval += bin_size

            last_thres = bin_list_i[-1]
            if last_thres != max_x:
                if last_thres > max_x:
                    bin_list_i[-1] = max_x
                else:
                    left = np.sum(np.logical_and(last_thres < x, x < max_x))
                    if left > (bin_size / 2):
                        bin_list_i.append(max_x)
                    else:
                        bin_list_i[-1] = max_x

            bin_list.append(bin_list_i)
        return bin_list

    def generate_label_to_lfs(self):
        label_to_lfs = {}
        label_to_idx_to_lfs = {}
        idx_to_lfs = defaultdict(list)
        for label in range(self.n_class):
            y = np.array(self.Y == label, dtype=np.int)
            min_acc = self.class_marginal[label] + self.min_acc_gain
            idx_to_lfs_i = {}
            for idx in range(self.n_feature):
                bin_list_i = self.bin_list[idx]
                x = self.X[:, idx]
                idx_lfs = self.generate_half_bounded_lf(x, y, idx, label, bin_list_i, min_acc) \
                          + self.generate_interval_lf(x, y, idx, label, bin_list_i, min_acc)
                if len(idx_lfs) > 1:
                    idx_to_lfs_i[idx] = idx_lfs
                    idx_to_lfs[idx] += idx_lfs
            lfs_for_label = list(chain.from_iterable(idx_to_lfs_i.values()))
            assert len(lfs_for_label) > 1, f'cannot find any lf for label {label}, please lower the min support or the min acc gain!'
            label_to_idx_to_lfs[label] = idx_to_lfs_i
            label_to_lfs[label] = list(chain.from_iterable(idx_to_lfs_i.values()))
        return label_to_lfs, idx_to_lfs, label_to_idx_to_lfs

    def generate_half_bounded_lf(self, x, y, idx, label, bin_list, min_acc):
        lfs = []
        n = len(x)
        for thres in bin_list[1:-1]:
            greater_then_idx = x > thres
            greater_acc = self.calc_acc(y[greater_then_idx])
            if greater_acc > min_acc and np.sum(greater_then_idx) > self.min_support:
                propensity = np.sum(greater_then_idx) / n
                e = GreaterExpression(idx=idx, threshold=thres)
                lf = LF(e=e, label=label, acc=greater_acc, propensity=propensity)
                lfs.append(lf)
            else:
                less_then_idx = x < thres
                less_acc = self.calc_acc(y[less_then_idx])
                if less_acc > min_acc and np.sum(less_then_idx) > self.min_support:
                    propensity = np.sum(less_then_idx) / n
                    e = LessExpression(idx=idx, threshold=thres)
                    lf = LF(e=e, label=label, acc=less_acc, propensity=propensity)
                    lfs.append(lf)
        return lfs

    def generate_interval_lf(self, x, y, idx, label, bin_list, min_acc):
        lfs = []
        n = len(x)
        for i in range(1, len(bin_list) - 1):
            thres = (bin_list[i], bin_list[i + 1])
            in_interval_idx = np.logical_and(thres[0] < x, x < thres[1])
            in_interval_acc = self.calc_acc(y[in_interval_idx])
            if in_interval_acc > min_acc and np.sum(in_interval_idx) > self.min_support:
                propensity = np.sum(in_interval_idx) / n
                e = InIntervalExpression(idx=idx, threshold=thres)
                lf = LF(e=e, label=label, acc=in_interval_acc, propensity=propensity)
                lfs.append(lf)
            else:
                out_interval_idx = np.logical_or(thres[0] > x, x > thres[1])
                out_interval_acc = self.calc_acc(y[out_interval_idx])
                if out_interval_acc > min_acc and np.sum(out_interval_acc) > self.min_support:
                    propensity = np.sum(out_interval_idx) / n
                    e = OutIntervalExpression(idx=idx, threshold=thres)
                    lf = LF(e=e, label=label, acc=out_interval_acc, propensity=propensity)
                    lfs.append(lf)
        return lfs

    def one_feature_one_lf_generate(self, n_lfs: Union[int, List[int]] = 10) -> FeatureLFApplier:
        if isinstance(n_lfs, int):
            try:
                sampled_idx = self.generator.choice(list(self.idx_to_lfs.keys()), size=n_lfs)
                lfs = [self.generator.choice(self.idx_to_lfs[idx]) for idx in sampled_idx]
            except ValueError as e:
                raise NoEnoughLFError()
        else:
            assert len(n_lfs) == self.n_class
            labels = list(range(self.n_class))
            lfs = []
            for label, n_lfs_i in zip(labels, n_lfs):
                idx_to_lf = self.label_to_idx_to_lfs[label]
                try:
                    sampled_idx = self.generator.choice(list(idx_to_lf.keys()), size=n_lfs_i)
                    lfs_i = [self.generator.choice(idx_to_lf[idx]) for idx in sampled_idx]
                except ValueError as e:
                    raise NoEnoughLFError(label)
                lfs += lfs_i
        return FeatureLFApplier(lfs)


class NGramLFGenerator(AbstractLFGenerator):
    def __init__(self,
                 dataset: TextDataset,
                 y: Optional[np.ndarray] = None,
                 vectorizer: CountVectorizer = None,
                 ngram_range=(1, 1),
                 min_acc_gain: float = 0.1,
                 min_support: float = 0.01,
                 random_state=None
                 ):

        super(NGramLFGenerator, self).__init__(dataset, y, min_acc_gain, min_support, random_state)
        if vectorizer is None:
            vectorizer = CountVectorizer(strip_accents='ascii',
                                         # stop_words='english',
                                         ngram_range=ngram_range,
                                         analyzer='word',
                                         max_df=0.90,
                                         min_df=self.min_support / self.n_data,
                                         max_features=None,
                                         vocabulary=None,
                                         binary=False)

        corpus = [i['text'] for i in self.dataset.examples]
        self.X = vectorizer.fit_transform(corpus)
        self.vectorizer = vectorizer
        self.idx_to_ngram = vectorizer.get_feature_names()
        self.n_feature = self.X.shape[1]
        self.label_to_candidate_lfs = self.generate_label_to_lfs()
        self.lf_applier_type = partial(NGramLFApplier, vectorizer=vectorizer)

    def generate_label_to_lfs(self):
        label_to_lfs = {}
        for label in range(self.n_class):
            y = np.array(self.Y == label, dtype=np.int)
            min_acc = self.class_marginal[label] + self.min_acc_gain
            lfs = []
            for idx in range(self.n_feature):
                x = self.X[:, idx].toarray().squeeze()
                exist_idx = x > 0
                exist_acc = self.calc_acc(y[exist_idx])
                if exist_acc > min_acc and np.sum(exist_idx) > self.min_support:
                    ngram = self.idx_to_ngram[idx]
                    propensity = np.sum(exist_idx) / self.n_data
                    e = NGramExpression(idx=idx, threshold=0, ngram=ngram)
                    lf = LF(e=e, label=label, acc=exist_acc, propensity=propensity)
                    lfs.append(lf)
            assert len(lfs) > 1, f'cannot find any lf for label {label}, please lower the min support or the min acc gain!'
            label_to_lfs[label] = lfs
        return label_to_lfs
