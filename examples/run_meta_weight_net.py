import logging
import torch
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.labelmodel import Snorkel
from wrench.metalearning import MetaWeightNet

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
train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    extract_feature=True,
    extract_fn='bert',  # extract bert embedding
    model_name='bert-base-cased',
    cache_name='bert'
)

#### Run label model: Snorkel
label_model = Snorkel(
    lr=0.01,
    l2=0.0,
    n_epochs=10
)
label_model.fit(
    dataset_train=train_data,
    dataset_valid=valid_data
)
acc = label_model.test(test_data, 'acc')
logger.info(f'label model test acc: {acc}')

#### Filter out uncovered training data
train_data = train_data.get_covered_subset()
aggregated_labels = label_model.predict(train_data)

#### Run end model: MetaWeightNet
model = MetaWeightNet(
    batch_size=128,
    test_batch_size=512,
    n_steps=10000,
    backbone='MLP',
    optimizer='SGD',
    optimizer_lr=1e-1,
    optimizer_weight_decay=5e-4,
    v_net_optimizer='Adam',
    v_net_optimizer_lr=1e-3,
    v_net_optimizer_weight_decay=1e-4,
)
model.fit(
    dataset_train=train_data,
    y_train=aggregated_labels,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='acc',
    patience=100,
    device=device
)
acc = model.test(test_data, 'acc')
logger.info(f'MetaWeightNet test acc: {acc}')

