import logging
import torch
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.labelmodel import EBCC, IBCC


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cuda')

#### Load dataset
dataset_path = '../datasets/'
data = 'agnews'
train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    extract_feature=True,
    extract_fn='bert',  # extract bert embedding
    model_name='bert-base-cased',
    cache_name='bert'
)
train_data_c = train_data.get_covered_subset()

#### Inference for EBCC and IBCC
logger.info('============inference============')
ebcc = EBCC(
    num_groups=5,
    repeat=100,
    inference_iter=100,
    empirical_prior=True,
)
ibcc = IBCC()

ebcc.fit(
    dataset_train=train_data_c
)
ibcc.fit(
    dataset_train=train_data_c
)

#### Test for EBCC and IBCC
logger.info('============test============')
ebcc.predict_proba(train_data_c)
acc_ebcc = ebcc.test(train_data_c, 'acc')
acc_test_ebcc = ebcc.test(test_data, 'acc')

acc_ibcc = ibcc.test(train_data_c, 'acc')
acc_test_ibcc = ibcc.test(test_data, 'acc')

logger.info(f'label model train/test acc on ebcc: {acc_ebcc}, {acc_test_ebcc}, seed={ebcc.params["seed"]}')
logger.info(f'label model train/test acc on ibcc: {acc_ibcc}, {acc_test_ibcc}')
