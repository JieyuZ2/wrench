import logging

from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.labelmodel import Snorkel
from wrench.seq_labelmodel import SeqLabelModelWrapper

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

#### Load dataset
dataset_path = '../datasets/'
data = 'laptopreview'
train_data, valid_data, test_data = load_dataset(dataset_path, data, extract_feature=False)

#### Run label model: Snorkel
label_model = SeqLabelModelWrapper(
    label_model_class=Snorkel,
    lr=0.01,
    l2=0.0,
    n_epochs=10
)
label_model.fit(dataset_train=train_data, dataset_valid=valid_data)
f1 = label_model.test(test_data, 'f1_seq')
logger.info(f'label model test f1: {f1}')

