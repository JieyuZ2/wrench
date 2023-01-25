import logging
import torch
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.labelmodel import HyperLM

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cpu')

#### Load dataset
dataset_path = '../datasets/'
data = 'agnews'
train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    extract_feature=False,
)
train_data_c = train_data.get_covered_subset()

#### Inference for HyperLM
logger.info('============inference============')
hlm = HyperLM()

#### Training for HyperLM
# This method does not require training

#### Test for HyperLM
logger.info('============test============')
acc_hlm = hlm.test(train_data_c, 'acc')
acc_test_hlm = hlm.test(test_data, 'acc')

logger.info(f'label model train/test acc on HyperLM: {acc_hlm}, {acc_test_hlm}')
