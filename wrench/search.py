import _thread as thread
import logging
import multiprocessing
import random
import sys
import threading
import warnings
from functools import partial
from typing import Any, Dict, Optional, Union, Callable

import optuna
from optuna.samplers import GridSampler
from optuna.trial import Trial
from tqdm.auto import tqdm, trange

from .basemodel import BaseModel
from .evaluation import metric_to_direction

logger = logging.getLogger(__name__)


def quit_function(fn_name):
    # raise Exception(f'[TIMEOUT] {fn_name} take too long!')
    print(f'[TIMEOUT] {fn_name} take too long!', file=sys.stderr)
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt


def exit_after(s):
    '''
    use as decorator to exit process if
    function takes longer than s seconds
    '''

    def outer(fn):
        def inner(*args, **kwargs):
            if s > 0:
                timer = threading.Timer(s, quit_function, args=[fn.__name__])
                timer.start()
                try:
                    result = fn(*args, **kwargs)
                finally:
                    timer.cancel()
                return result
            else:
                return fn(*args, **kwargs)

        return inner

    return outer


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
                dataset_train,
                dataset_valid,
                y_train=None,
                y_valid=None,
                process_fn: Callable = single_process,
                metric: Optional[Union[str, Callable]] = 'f1_macro',
                direction: Optional[str] = 'auto',
                n_repeats: Optional[int] = 1,
                n_trials: Optional[int] = 100,
                n_jobs: Optional[int] = 1,
                trial_timeout: Optional[int] = -1,
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
        if trial_timeout > 0: warnings.warn('Parallel searching does not support trial time out!')
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

        @exit_after(trial_timeout)
        def objective(trial):
            try:
                suggestions = fetch_hyperparas_suggestions(search_space, trial)
                metric_value = 0
                for i in trange(n_repeats):
                    metric_value += worker((suggestions, i))
                value = metric_value / n_repeats
                return value
            except KeyboardInterrupt:
                raise Exception('[KeyboardInterrupt] may due to timeout')

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, catch=(Exception,))

    logger.info(f'[END: BEST VAL / PARAMS] Best value: {study.best_value}, Best paras: {study.best_params}')
    return study.best_params
