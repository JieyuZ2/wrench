import logging
from pingouin import distance_corr
from wrench._logging import LoggingHandler
from wrench.dataset import load_dataset
import numpy as np

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

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

X = train_data.features
Y = np.array(train_data.labels)
L = np.array(train_data.weak_labels)

for i in range(train_data.n_lf):
    mu_i = np.zeros((train_data.n_class, train_data.n_class+1))
    for j, y in enumerate(Y):
        if L[j, i] == -1:
            mu_i[y, -1] += 1
        else:
            mu_i[y, L[j, i]] += 1



y_dcor, _ = distance_corr(X, Y)
logger.info(f'(x, y) distance correlation: {y_dcor:.3f}')

lf_dcors = []
for i in range(train_data.n_lf):
    l = L[:, i]
    correct = np.where(l==Y, 1, 0)
    non_abs = l!=-1
    if np.all(correct[non_abs]==correct[non_abs][0]):
        dcor = 0
    else:
        dcor, pval = distance_corr(X[non_abs], correct[non_abs])
    logger.info(f'(x, lf {i}) distance correlation: {dcor:.3f}')
    lf_dcors.append(dcor)

logger.info(f'(x, lf) distance correlation (mean): {np.mean(lf_dcors):.3f}')

a=1