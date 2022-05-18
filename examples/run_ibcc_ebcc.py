import copy
import logging
import torch
from wrench.dataset import load_dataset, TextDataset
from wrench._logging import LoggingHandler
from wrench.labelmodel import EBCC, IBCC, Snorkel


def concat(d1: TextDataset, d2: TextDataset) -> TextDataset:
    dataset = TextDataset()
    dataset.ids = d1.ids + d2.ids
    dataset.labels = d1.labels + d2.labels
    dataset.examples = d1.examples + d2.examples
    dataset.weak_labels = d1.weak_labels + d2.weak_labels
    dataset.n_class = d1.n_class
    dataset.n_lf = d1.n_lf

    return dataset


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
train_data_c = train_data.get_covered_subset()
#### Run label model: Snorkel

# print('============inductive============')
# label_model_generate = EBCC(
#     num_groups=5,
#     inference_iter=1,
#     max_iter=1,
#     empirical_prior=True,
#     kernel_function=RBF(length_scale=1)
# )
#

# label_model_generate.seed = 12345
# train_data_c.labels = probs_to_preds(label_model_generate.predict_proba(train_data_c))

print('============inference============')
ebcc = EBCC(
    num_groups=5,
    repeat=10,
    inference_iter=100,
    empirical_prior=True,
)
ibcc = IBCC()
snorkel = Snorkel(
    lr=0.01,
    l2=0.0,
    n_epochs=10
)

ebcc.fit(
    dataset_train=train_data_c
)
snorkel.fit(
    dataset_train=train_data_c,
    dataset_valid=valid_data
)

print('============test============')
ebcc.predict_proba(train_data_c)
acc_ebcc = ebcc.test(train_data_c, 'acc')
acc_test_ebcc = ebcc.test(test_data, 'acc')

acc_ibcc = ibcc.test(train_data_c, 'acc')
acc_test_ibcc = ibcc.test(test_data, 'acc')

acc_s = snorkel.test(train_data_c, 'acc')
acc_test_s = snorkel.test(test_data, 'acc')

logger.info(f'label model train/test acc on gp-ebcc: {acc_ebcc}, {acc_test_ebcc}, seed={ebcc.seed}')
logger.info(f'label model train/test acc on gp-ibcc: {acc_ibcc}, {acc_test_ibcc}')
logger.info(f'label model train/test acc on snorkel: {acc_s}, {acc_test_s}')
