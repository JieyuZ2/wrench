import logging
import torch
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.classification import WeaSEL

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


#### Run WeaSEL
model = WeaSEL(
    temperature=0.33,
    dropout=0.3,
    hidden_size=512,

    batch_size=32,
    real_batch_size=-1,
    test_batch_size=128,
    n_steps=100000,
    grad_norm=-1,

    backbone='MLP',
    backbone_dropout=0.2,  # fine  tune all
    backbone_hidden_size=256,
    backbone_n_hidden_layers=2,

    use_lr_scheduler=True,
    optimizer='default',
    optimizer_lr=0.001,
    optimizer_weight_decay=0.0,
)
model.fit(
    dataset_train=train_data,
    dataset_valid=valid_data,
    evaluation_step=5,
    metric='f1_binary',
    patience=200,
    device=device
)
f1 = model.test(test_data, 'f1_binary')
logger.info(f'WeaSEL test f1: {f1}')
a=1

