import abc
from copy import deepcopy
from typing import Union

import numpy as np


def get_search_space(model: Union[str, abc.ABCMeta]):
    if isinstance(model, abc.ABCMeta):
        model = model.__name__
    search_space = SEARCH_SPACE[model]
    filter_fn = None
    if model == 'Denoise':
        filter_fn = filter_fn_for_denoise
    return search_space, filter_fn


# filter function (input to grid_search) for Denoise to filter invalid search grid,
def filter_fn_for_denoise(grids, para_names):
    c1, c2 = para_names.index('c1'), para_names.index('c2')
    return [grid for grid in grids if grid[c1] + grid[c2] < 1]


SEARCH_SPACE = {
    'Snorkel'            : {
        'lr'      : np.logspace(-5, -1, num=5, base=10),
        'l2'      : np.logspace(-5, -1, num=5, base=10),
        'n_epochs': [5, 10, 50, 100, 200],
    },
    'GenerativeModel'    : {
        'lr'      : [1e-4, 1e-5, 5e-5],
        'l2'      : np.logspace(-5, -1, num=5, base=10),
        'n_epochs': [5, 10, 50, 100, 200],
    },

    'LogRegModel'        : {
        'lr'        : np.logspace(-5, -1, num=5, base=10),
        'l2'        : np.logspace(-5, -1, num=5, base=10),
        'batch_size': [32, 128, 512],
    },
    'MLPModel'           : {
        'lr'         : np.logspace(-5, -1, num=5, base=10),
        'l2'         : np.logspace(-5, -1, num=5, base=10),
        'batch_size' : [32, 128, 512],
        'hidden_size': [100],
        'dropout'    : [0.0],
    },
    'BertClassifierModel': {
        'lr'        : [5e-5, 3e-5, 2e-5],
        'batch_size': [32, 16],
    },
    'Cosine'             : {
        'lr'            : [1e-5, 1e-6],
        'l2'            : [1e-4],
        'batch_size'    : [32],
        'teacher_update': [50, 100, 200],
        'lamda'         : [0.01, 0.05, 0.1],
        'thresh'        : [0.2, 0.4, 0.6, 0.8],
        'margin'        : [1.0],
        'mu'            : [1.0],
    },

    'Denoise'            : {
        'lr'        : np.logspace(-4, -2, num=3, base=10),
        'batch_size': [32, 128, 512],
        'c1'        : [0.1, 0.3, 0.5, 0.7, 0.9],
        'c2'        : [0.1, 0.3, 0.5, 0.7, 0.9],
    },

    'CHMM'               : {
        'nn_lr'                 : [1e-3, 5e-4, 1e-4],
        'hmm_lr'                : [1e-2, 5e-3, 1e-3],
        'batch_size'            : [16, 64, 128],
        'num_nn_pretrain_epochs': [2, 5],
        'num_train_epochs'      : [50],
    },
    'HMM'                : {
        'redundancy_factor': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
        'n_epochs'         : [50],
    },

    'LSTMTaggerModel'    : {
        'lr'                    : [1e-3, 5e-3, 1e-2],
        'l2'                    : [1e-8],
        'batch_size'            : [64, 32, 16],
        'dropout'               : [0.0, 0.5],

        'word_hidden_dim'       : [200],
        'word_feature_extractor': ['GRU', 'LSTM'],
        'n_word_hidden_layer'   : [1],

        'use_char'              : [True],
        'char_emb_dim'          : [30],
        'char_hidden_dim'       : [50],
        'char_feature_extractor': ['CNN'],

    },
    'BERTTaggerModel'    : {
        'lr'        : [5e-5, 3e-5, 2e-5],
        'lr_crf'    : [1e-3, 5e-3, 1e-2],
        'l2'        : [1e-6],
        'l2_crf'    : [1e-8],
        'batch_size': [32, 16, 8],
    },

}

SEARCH_SPACE['LSTMConNetModel'] = deepcopy(SEARCH_SPACE['LSTMTaggerModel'])
SEARCH_SPACE['LSTMConNetModel'].update({'n_steps_phase1': [200, 500, 1000]})
SEARCH_SPACE['BERTConNetModel'] = deepcopy(SEARCH_SPACE['BERTTaggerModel'])
SEARCH_SPACE['BERTConNetModel'].update({'n_steps_phase1': [200, 500, 1000]})
