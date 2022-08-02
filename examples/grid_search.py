import logging
import torch
import numpy as np
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.search import grid_search
from wrench.endmodel import EndClassifierModel

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cuda')

if __name__ == '__main__':
    #### Load dataset
    dataset_path = '../datasets/'
    data = 'youtube'
    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        data,
        extract_feature=True,
        extract_fn='bert', # extract bert embedding
        model_name='bert-base-cased',
        cache_name='bert'
    )

    #### Search Space
    search_space = {
        'optimizer_lr': np.logspace(-5, -1, num=5, base=10),
        'optimizer_weight_decay': np.logspace(-5, -1, num=5, base=10),
    }

    #### Initialize the model: MLP
    model = EndClassifierModel(
        batch_size=128,
        test_batch_size=512,
        n_steps=10000,
        backbone='MLP',
        optimizer='Adam',
    )

    #### Search best hyper-parameters using validation set in parallel
    n_trials = 100
    n_repeats = 3
    searched_paras = grid_search(
        model,
        dataset_train=train_data,
        dataset_valid=valid_data,
        metric='acc',
        direction='auto',
        search_space=search_space,
        n_repeats=n_repeats,
        n_trials=n_trials,
        parallel=True,
        device=device,
    )


    #### Run end model: MLP
    model = EndClassifierModel(
        batch_size=128,
        test_batch_size=512,
        n_steps=10000,
        backbone='MLP',
        optimizer='Adam',
        **searched_paras
    )
    model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data,
        evaluation_step=10,
        metric='acc',
        patience=100,
        device=device
    )
    acc = model.test(test_data, 'acc')
    logger.info(f'end model (MLP) test acc: {acc}')

