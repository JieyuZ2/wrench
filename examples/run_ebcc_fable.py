import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wrench.dataset import load_dataset, BaseDataset, get_dataset_type, TorchDataset
from wrench.logging import LoggingHandler
from wrench.labelmodel import Fable, EBCC
from sklearn.gaussian_process.kernels import PairwiseKernel


def concat(d1: BaseDataset, d2: BaseDataset, name: str) -> BaseDataset:
    dataset = get_dataset_type(name)()
    dataset.ids = d1.ids + d2.ids
    dataset.labels = d1.labels + d2.labels
    dataset.examples = d1.examples + d2.examples
    dataset.weak_labels = d1.weak_labels + d2.weak_labels
    dataset.n_class = d1.n_class
    dataset.n_lf = d1.n_lf
    dataset.features = np.vstack([d1.features, d2.features])

    return dataset


def create_dataset(batch: dict, dataset: BaseDataset, name: str) -> BaseDataset:
    new_set = get_dataset_type(name)()

    new_set.ids = batch['ids'].tolist()
    new_set.labels = batch['labels'].tolist()
    new_set.examples = batch['data']
    new_set.weak_labels = batch['weak_labels'].tolist()
    new_set.n_class = dataset.n_class
    new_set.n_lf = dataset.n_lf
    new_set.features = batch['features']

    return new_set


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)
device = torch.device('cuda:0')

#### Load dataset
num_group = 3
datalist = ['cdr']
num_correct = 1000

### Transductive test
print('============test (transductive)============')
dataset_path = '../datasets/'
data = 'commercial'
metric = 'f1_binary'

train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    extract_feature=True,
    extract_fn='bert',
    model_name='roberta-base',  # roberta-base, roberta; bert-base-uncased, bert
    cache_name='roberta',
    normalize=True,
    device=device
)

### Create dataset for transductive test
all_dataset = concat(train_data, valid_data, data)
all_dataset = concat(all_dataset, test_data, data)
train = all_dataset.get_covered_subset()

### Initialize label model
fable = Fable(
    num_groups=num_group,
    inference_iter=10,
    a_v=all_dataset.n_class * num_group * num_correct,
    b_v=1,
    empirical_prior=True,
    kernel_function=PairwiseKernel('cosine'),
    desired_rank=50,
    device=device
)
ebcc = EBCC()

### Test Fable and EBCC
ebcc_res = ebcc.test(all_dataset, metric)
if len(all_dataset.ids) > 20000:
    res = []
    for batch in tqdm(DataLoader(TorchDataset(all_dataset), shuffle=True, batch_size=10000)):
        batch_set = create_dataset(batch, all_dataset, data)
        res.append(fable.test(batch_set, metric, batch_learning=True))
    fable_res = np.mean(res)
else:
    fable_res = fable.test(all_dataset, metric)

logger.info(f'label model acc on EBCC: {ebcc_res}')
logger.info(f'label model acc on FABLE: {fable_res}')

