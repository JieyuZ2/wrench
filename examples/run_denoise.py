import logging
import torch
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.classification import Denoise

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cuda')

#### Load dataset
dataset_path = '../datasets/'
data = 'youtube'
bert_model_name = 'bert-base-cased'
train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    extract_feature=True,
    extract_fn='bert', # extract bert embedding
    model_name=bert_model_name,
    cache_name='bert'
)


#### Run denoise
model = Denoise(
    alpha=0.6,
    c1=0.2,
    c2=0.7,
    hidden_size=100,

    batch_size=16,
    real_batch_size=8,
    test_batch_size=128,
    n_steps=10000,
    grad_norm=1.0,

    # backbone='MLP',
    backbone='BERT',
    backbone_model_name=bert_model_name,
    backbone_fine_tune_layers=-1,  # fine  tune all
    optimizer='AdamW',
    optimizer_lr=5e-5,
    optimizer_weight_decay=0.0,

    label_model='Snorkel',
    label_model_n_epochs=10,
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
logger.info(f'Denoise test acc: {acc}')

