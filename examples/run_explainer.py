import logging

import numpy as np
import torch

from wrench._logging import LoggingHandler
from wrench.dataset import load_dataset
from wrench.endmodel import EndClassifierModel
from wrench.explainer import Explainer, modify_training_labels
from wrench.labelmodel import Snorkel

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
train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    extract_feature=True
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
L = np.array(train_data.weak_labels)
aggregated_soft_labels = label_model.predict_proba(train_data)

#### Run end model
model = EndClassifierModel(
    batch_size=128,
    test_batch_size=512,
    n_steps=10000,
    backbone='LogReg',
    optimizer='Adam',
    optimizer_lr=1e-2,
    optimizer_weight_decay=0.0,
)
model.fit(
    dataset_train=train_data,
    y_train=aggregated_soft_labels,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='acc',
    patience=100,
    device=device
)
acc = model.test(test_data, 'acc')
logger.info(f'end model (LogReg) test acc: {acc}')

#### Compute IF score
explainer = Explainer(train_data.n_lf, train_data.n_class)
approx_w = explainer.approximate_label_model(L, aggregated_soft_labels)

if_type = 'if'  # or 'sif' or 'relatif'
mode = 'RW'  # or 'WM' or 'normal'
lr, weight_decay, epochs, batch_size = 0.01, 0.0, 1000, 128

IF_score = explainer.compute_IF_score(
    L, np.array(train_data.features), np.array(valid_data.features), np.array(valid_data.labels),
    if_type=if_type, mode=mode,
    lr=lr, weight_decay=weight_decay, epochs=epochs, batch_size=batch_size,
    device=device
)

alpha = 0.8 # sample 80%

modified_soft_labels = modify_training_labels(aggregated_soft_labels, L, approx_w, IF_score, alpha, sample_method='weight', normal_if=False, act_func='identity')

#### Run end model again
model = EndClassifierModel(
    batch_size=128,
    test_batch_size=512,
    n_steps=10000,
    backbone='LogReg',
    optimizer='Adam',
    optimizer_lr=1e-2,
    optimizer_weight_decay=0.0,
)
model.fit(
    dataset_train=train_data,
    y_train=modified_soft_labels,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='acc',
    patience=100,
    device=device
)
acc = model.test(test_data, 'acc')
logger.info(f'end model (LogReg) test acc: {acc}')
