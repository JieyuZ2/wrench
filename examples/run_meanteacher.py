import logging
import torch

from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.semisupervisedlearning import MeanTeacher

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cuda')

#### Load dataset
dataset_path = f'../datasets/'
data = 'youtube'
train_data, valid_data, test_data = load_dataset(dataset_path, data, extract_feature=True, extract_fn='bert', cache_name='bert')

#### Load given labeled data ids
idx, _ = train_data.load_labeled_ids_and_lf_exemplars(f'{dataset_path}/{data}/labeled_ids.json')

#### Run end model: MeanTeacher
model = MeanTeacher(
    batch_size=128,
    test_batch_size=512,
    n_steps=10000,
    backbone='MLP',
    optimizer='default',
    optimizer_lr=1e-3,
    optimizer_weight_decay=5e-4,
    use_lr_scheduler=True,
    lr_scheduler='default'
)
model.fit(
    dataset_train=train_data,
    labeled_data_idx=idx,
    dataset_valid=valid_data,
    evaluation_step=5,
    metric='f1_macro',
    patience=100,
    device=device
)
f1_macro = model.test(test_data, 'f1_macro')
logger.info(f'MeanTeacher test f1: {f1_macro}')
