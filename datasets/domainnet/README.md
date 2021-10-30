
## Using DomainNet Dataset to Construct Weak Supervision Tasks (Image Classification)

The overall procedure is based on this [paper](http://cs.brown.edu/people/sbach/files/mazzetto-icml21.pdf)

[1] Download the [DomainNet](http://ai.bu.edu/M3SDA/) datasets and unzip the 6 datasets under this folder: 

```
datasets 
     -- domainnet
         |-- clipart
         |-- infograph
         |-- painting
         |-- quickdraw
         |-- real
         |-- sketch
```


[2] Run the script to generate dataset:
```
python generate_dataset.py --data_root ./ --save_root ../ --target_domain real --n_classes 5 --n_labeler_per_domain 1 
--model_name resnet18  --lr 0.005 --n_epochs 100 --batch_size 64

```
It will set `real` domain as target domain, and use images from remaining domains to train weak labelers.
Finally, a new dataset folder `domainnet-real` will show up:
```
datasets 
     -- domainnet-real
         |-- train.json
         |-- valid.json
         |-- test.json
         |-- label.json
         |-- meta.json
     -- domainnet
         |-- clipart
         |-- infograph
         |-- painting
         |-- quickdraw
         |-- real
         |-- sketch
```

[2] Run the weak supervision method on the generated `domainnet-real` dataset:

```python
import logging
from pathlib import Path

import torch

from wrench.dataset import load_image_dataset
from wrench.endmodel import EndClassifierModel
from wrench.labelmodel import Snorkel
from wrench.logging import LoggingHandler
from wrench.classification import WeaSEL

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cuda')

#### Load dataset
dataset_path = Path('../datasets/')
image_root = dataset_path / 'domainnet'
data = 'domainnet-real'
train_data, valid_data, test_data = load_image_dataset(data_home=dataset_path, dataset=data, image_root_path=image_root, preload_image=True, extract_feature=True)

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
aggregated_hard_labels = label_model.predict(train_data)
aggregated_soft_labels = label_model.predict_proba(train_data)

#### Run end model: MLP
model = EndClassifierModel(
    batch_size=128,
    test_batch_size=512,
    n_steps=10000,
    backbone='MLP',
    optimizer='Adam',
    optimizer_lr=1e-2,
    optimizer_weight_decay=0.0,
)
model.fit(
    dataset_train=train_data,
    y_train=aggregated_hard_labels,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='acc',
    patience=100,
    device=device
)
acc = model.test(test_data, 'acc')
logger.info(f'end model (MLP) test acc: {acc}')

#### Run end model: ResNet
model = EndClassifierModel(
    batch_size=128,
    test_batch_size=512,
    n_steps=10000,
    backbone='ImageClassifier',
    backbone_model_name='resnet18',
    optimizer='Adam',
    optimizer_lr=1e-2,
    optimizer_weight_decay=0.0,
)
model.fit(
    dataset_train=train_data,
    y_train=aggregated_hard_labels,
    dataset_valid=valid_data,
    evaluation_step=10,
    metric='acc',
    patience=100,
    device=device
)
acc = model.test(test_data, 'acc')
logger.info(f'end model (ResNet) test acc: {acc}')


#### Run WeaSEL
model = WeaSEL(
    temperature=1.0,
    dropout=0.3,
    hidden_size=100,

    batch_size=16,
    real_batch_size=8,
    test_batch_size=128,
    n_steps=10000,
    grad_norm=1.0,

    backbone='ImageClassifier',
    backbone_model_name='resnet18',
    optimizer='AdamW',
    optimizer_lr=5e-5,
    optimizer_weight_decay=0.0,
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
logger.info(f'WeaSEL test acc: {acc}')

```

