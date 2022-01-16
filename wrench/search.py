import _thread as thread
import json
import logging
import multiprocessing
import random
import sys
import threading
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable

import optuna
from optuna.samplers import GridSampler
from optuna.trial import Trial
from tqdm.auto import tqdm, trange

from .basemodel import BaseModel
from .evaluation import metric_to_direction

logger = logging.getLogger(__name__)


def quit_function(fn_name):
    print(f'[TIMEOUT] {fn_name} take too long!', file=sys.stderr)
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt


def exit_after(s):
    '''
    use as decorator to exit process if function takes longer than s seconds
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


class StopWhenNotImproved:
    def __init__(self, patience: int, min_trials: int):
        self.patience = patience
        self.min_trials = min_trials
        self.no_improve_cnt = 0
        self.trial_cnt = 0
        self.best_value = None

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            self.trial_cnt += 1
            current_value = trial.value
            if self.best_value is None:
                self.best_value = current_value
            else:
                if current_value > self.best_value:
                    self.best_value = current_value
                    self.no_improve_cnt = 0
                else:
                    if self.trial_cnt > self.min_trials:
                        self.no_improve_cnt += 1
                        if self.no_improve_cnt >= self.patience:
                            study.stop()


class RecordCallback:
    def __init__(self, metric: str, save_path: str):
        self.metric = metric
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.record = {}
        self.trial_cnt = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            self.record[self.trial_cnt] = trial.params
            self.record[self.trial_cnt][self.metric] = trial.value
            self.trial_cnt += 1
            json.dump(self.record, open(self.save_path / 'search_record.json', 'w'), indent=4)


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
    m = deepcopy(model)
    m.fit(dataset_train=dataset_train, y_train=y_train, dataset_valid=dataset_valid, y_valid=y_valid,
          verbose=True, metric=metric, direction=direction, **suggestions, **kwargs)
    value = m.test(dataset_valid, metric_fn=metric)
    return value


def single_process_with_y_train(item, model, dataset_train, y_train, dataset_valid, y_valid, metric, direction, kwargs):
    suggestions, i = item
    kwargs = kwargs.copy()
    y_train_l = kwargs.pop('y_train_l')
    y_train = y_train_l[i]
    m = deepcopy(model)
    m.fit(dataset_train=dataset_train, y_train=y_train, dataset_valid=dataset_valid, y_valid=y_valid,
          verbose=False, metric=metric, direction=direction, **suggestions, **kwargs)
    value = m.test(dataset_valid, metric_fn=metric)
    return value


def single_process_with_seed(item, model, dataset_train, y_train, dataset_valid, y_valid, metric, direction, kwargs):
    suggestions, i = item
    kwargs = kwargs.copy()
    seeds = kwargs.pop('seeds')
    seed = seeds[i]
    m = deepcopy(model)
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
                filter_fn: Optional[Callable] = None,
                metric: Optional[Union[str, Callable]] = 'f1_macro',
                direction: Optional[str] = 'auto',
                n_repeats: Optional[int] = 1,
                n_trials: Optional[int] = 100,
                n_jobs: Optional[int] = 1,
                min_trials: Optional[Union[int, float]] = -1,
                study_patience: Optional[Union[int, float]] = -1,
                prune_threshold: Optional[float] = -1,
                trial_timeout: Optional[int] = -1,
                study_timeout: Optional[int] = None,
                parallel: Optional[bool] = False,
                study_name: Optional[str] = None,
                save_path: Optional[str] = None,
                **kwargs: Any):
    if direction == 'auto':
        direction = metric_to_direction(metric)
    worker = partial(process_fn,
                     model=model,
                     dataset_train=dataset_train,
                     y_train=y_train,
                     dataset_valid=dataset_valid,
                     y_valid=y_valid,
                     metric=metric,
                     direction=direction,
                     kwargs=kwargs)
    study = optuna.create_study(
        study_name=study_name,
        sampler=RandomGridSampler(search_space, filter_fn=filter_fn),
        direction=direction,
    )

    n_grids = len(study.sampler._all_grids)
    if isinstance(min_trials, float):
        min_trials = int(min_trials * n_grids)
    if isinstance(study_patience, float):
        study_patience = int(study_patience * n_grids)

    callbacks = []
    if study_patience > 0:
        callbacks.append(StopWhenNotImproved(patience=study_patience, min_trials=min_trials))
    if save_path is not None:
        callbacks.append(RecordCallback(metric=metric, save_path=save_path))

    if parallel:
        if trial_timeout > 0: warnings.warn('Parallel searching does not support trial time out!')
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(n_repeats)

        def parallel_objective(trial: Trial):
            suggestions = fetch_hyperparas_suggestions(search_space, trial)
            metric_value = 0
            for val in tqdm(pool.imap_unordered(worker, [(suggestions, i) for i in range(n_repeats)]), total=n_repeats):
                metric_value += val
            value = metric_value / n_repeats
            return value

        study.optimize(parallel_objective, n_trials=n_trials, n_jobs=n_jobs, catch=(Exception,), callbacks=callbacks, timeout=study_timeout)

    else:

        @exit_after(trial_timeout)
        def objective(trial: Trial):
            try:
                suggestions = fetch_hyperparas_suggestions(search_space, trial)
                metric_value = 0
                for i in trange(n_repeats):
                    val = worker((suggestions, i))
                    metric_value += val
                    if prune_threshold > 0 and trial._trial_id > 0:
                        if (trial.study.best_value - val) > (prune_threshold * trial.study.best_value):
                            return metric_value / (i + 1)
                value = metric_value / n_repeats
                return value
            except KeyboardInterrupt:
                raise Exception('[KeyboardInterrupt] may due to timeout')

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, catch=(Exception,), callbacks=callbacks, timeout=study_timeout)

    logger.info(f'[END: BEST VAL / PARAMS] Best value: {study.best_value}, Best paras: {study.best_params}')
    return study.best_params
