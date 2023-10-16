import logging
import torch
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.classification import SepLL

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cuda')

#### Load dataset
dataset_path = '../datasets/'
data = 'census'
bert_model_name = 'bert-base-cased'
train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    extract_feature=True,
    extract_fn='bert', # extract bert embedding
    model_name=bert_model_name,
    cache_name='bert'
)


#### Run SepLL
model = SepLL(
    batch_size=32,
    real_batch_size=-1,
    test_batch_size=128,
    # grad_norm=1.0,

    backbone='MLP',
    # backbone='BERT',
    # backbone_model_name=bert_model_name,
    # optimizer='AdamW',
    # optimizer_lr=3e-5,
    # optimizer_weight_decay=0.0,

    # SepLL specific
    add_unlabeled=False,
    class_noise=0.0,
    lf_l2_regularization=0.1,
)
model.fit(
    dataset_train=train_data,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='f1_binary',
    patience=100,
    device=device
)
f1 = model.test(test_data, 'f1_binary')
logger.info(f'SepLL test f1: {f1}')

