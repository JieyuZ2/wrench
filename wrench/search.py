from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import multiprocessing
from functools import partial
import random
from tqdm import tqdm
import numpy as np

import optuna
from optuna.trial import Trial
from optuna.samplers import GridSampler

from .basemodel import BaseModel
from .dataset import BaseDataset
from .evaluation import metric_to_direction

logger = logging.getLogger(__name__)


class RandomGridSampler(GridSampler):
    def __init__(self, search_space, filter_fn: Optional[Callable] = None) -> None:
        super().__init__(search_space=search_space)
        if filter_fn is not None:
            self._all_grids = filter_fn(self._all_grids, self._param_names)
            self._n_min_trials = len(self._all_grids)
        random.shuffle(self._all_grids)
        self.search_log = {}


def fetch_hyperparas_suggestions(search_space: Dict, trial: Trial):
    return {key: trial.suggest_categorical(key, search_space[key]) for key in search_space}


def single_process(item, model, dataset_train, y_train, dataset_valid, y_valid, metric, direction, kwargs):
    suggestions, i = item
    kwargs = kwargs.copy()
    hyperparas = model.hyperparas
    m = model.__class__(**hyperparas)
    m.fit(dataset_train=dataset_train, y_train=y_train, dataset_valid=dataset_valid, y_valid=y_valid,
                    verbose=False, metric=metric, direction=direction, **suggestions, **kwargs)
    value = m.test(dataset_valid, metric_fn=metric)
    return value


def single_process_with_y_train(item, model, dataset_train, y_train, dataset_valid, y_valid, metric, direction, kwargs):
    suggestions, i = item
    kwargs = kwargs.copy()
    y_train_l = kwargs.pop('y_train_l')
    y_train = y_train_l[i]
    hyperparas = model.hyperparas
    m = model.__class__(**hyperparas)
    m.fit(dataset_train=dataset_train, y_train=y_train, dataset_valid=dataset_valid, y_valid=y_valid,
                    verbose=False, metric=metric, direction=direction, **suggestions, **kwargs)
    value = m.test(dataset_valid, metric_fn=metric)
    return value


def single_process_with_seed(item, model, dataset_train, y_train, dataset_valid, y_valid, metric, direction, kwargs):
    suggestions, i = item
    kwargs = kwargs.copy()
    seeds = kwargs.pop('seeds')
    seed = seeds[i]
    hyperparas = model.hyperparas
    m = model.__class__(**hyperparas)
    m.fit(dataset_train=dataset_train, y_train=y_train, dataset_valid=dataset_valid, y_valid=y_valid,
                    verbose=False, metric=metric, direction=direction, seed=seed, **suggestions, **kwargs)
    value = m.test(dataset_valid, metric_fn=metric)
    return value


def grid_search(model: BaseModel,
                search_space: Dict,
                dataset_train: Union[BaseDataset, np.ndarray],
                dataset_valid: Union[BaseDataset, np.ndarray],
                y_train: Optional[np.ndarray] = None,
                y_valid: Optional[np.ndarray] = None,
                process_fn: Callable = single_process,
                metric: Optional[Union[str, Callable]] = 'f1_macro',
                direction: Optional[str] = 'auto',
                n_repeats: Optional[int] = 1,
                n_trials: Optional[int] = 100,
                n_jobs: Optional[int] = 1,
                parallel: Optional[bool] = False,
                filter_fn: Optional[Callable] = None,
                study_name: Optional[str] = None,
                **kwargs: Any):

    if direction == 'auto':
        direction = metric_to_direction(metric)
    worker = partial(process_fn, model=model, dataset_train=dataset_train, y_train=y_train,
                     dataset_valid=dataset_valid, y_valid=y_valid, metric=metric, direction=direction, kwargs=kwargs)
    study = optuna.create_study(study_name=study_name, sampler=RandomGridSampler(search_space, filter_fn=filter_fn), direction=direction)

    if parallel:
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(n_repeats)

        def parallel_objective(trial):
            suggestions = fetch_hyperparas_suggestions(search_space, trial)
            metric_value = 0
            for val in tqdm(pool.imap_unordered(worker, [(suggestions, i) for i in range(n_repeats)]), total=n_repeats):
                metric_value += val
            value = metric_value / n_repeats
            return value

        study.optimize(parallel_objective, n_trials=n_trials, n_jobs=n_jobs, catch=(Exception,))

    else:
        def objective(trial):
            suggestions = fetch_hyperparas_suggestions(search_space, trial)
            metric_value = 0
            for i in range(n_repeats):
                metric_value += worker((suggestions, i))
            value = metric_value / n_repeats
            return value

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, catch=(Exception, ))

    logger.info(f'[END: BEST VAL / PARAMS] Best value: {study.best_value}, Best paras: {study.best_params}')
    return study.best_params