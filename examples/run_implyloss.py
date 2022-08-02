import logging
import torch

from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.classification import ImplyLoss

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cuda')

#### Load dataset
dataset_path = f'../datasets/'
data = 'census'
train_data, valid_data, test_data = load_dataset(dataset_path, data, extract_feature=True)

#### Load given labeled data ids
idx, exemplars = train_data.load_labeled_ids_and_lf_exemplars(f'{dataset_path}/{data}/labeled_ids.json')

#### Run end model: ImplyLoss
model = ImplyLoss(
    hidden_size=100,
    q=0.2,
    gamma=0.1,

    batch_size=128,
    test_batch_size=512,
    n_steps=10000,
    backbone='MLP',
    optimizer='SGD',
    optimizer_lr=1e-1,
    optimizer_weight_decay=5e-4,
)
model.fit(
    dataset_train=train_data,
    labeled_data_idx=idx,
    exemplar_idx=exemplars,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='f1_macro',
    patience=100,
    device=device
)
f1_macro = model.test(test_data, 'f1_macro')
logger.info(f'ImplyLoss test f1: {f1_macro}')
