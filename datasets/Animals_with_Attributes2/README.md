
## Using Animals-with-Attributes2 Dataset to Construct Weak Supervision Tasks (Image Classification)

The overall procedure is based on this [paper](http://proceedings.mlr.press/v130/mazzetto21a/mazzetto21a.pdf).
This dataset contains attributes for animals and splits the classes into seen/unseen classes.
We use seen classes to train attribute detector, and then use the trained attribute detector and the attributes of unseen classes to build weak labelers for unseen classes.
We could construct weak supervision tasks for each pair of unseen classes.

### Prepare attribute detector

#### Option 1

Download the preprocessed data without training attribute detectors (the `Animals_with_Attributes2.zip` in this [link](https://huggingface.co/datasets/jieyuz2/WRENCH/tree/main)).

Unzip the data under this folder: 

```
datasets 
     -- Animals_with_Attributes2
         |-- JPEGImages
         |-- preprocessed_aa2.pkl
```

#### Option 2

[1] Download the 13 GB [Animals_with_Attributes2](https://cvml.ist.ac.at/AwA2/) datasets and unzip it under this folder: 

```
datasets 
     -- Animals_with_Attributes2
         |-- JPEGImages
         |-- classes.txt
         |-- testclasses.txt
         |-- trainclasses.txt
         |-- predicates.txt
         |-- predicate-matrix-binary.txt
```

We only show necessary files.

[2] Run the script to train attribute detectors:
```
python preprocess.py --data_root ./ --model_name resnet18  --lr 0.005 --n_epochs 5 --batch_size 512
```

It will train the 85 attribute detectors and produce a `preprocessed_aa2.pkl` under the `Animals_with_Attributes2` folder.

### Build weak supervision task

[1] Pick two unseen classes from ['chimpanzee', 'giant+panda', 'leopard', 'persian+cat', 'pig', 'hippopotamus', 'humpback+whale', 'raccoon', 'rat', 'seal']

[2] Run the script to generate dataset:
```
python generate_dataset.py --path ./preprocessed_aa2.pkl --save_root ../ --threshold 0.5 --class1 chimpanzee --class2 seal 
```
It will generate a new data folder named `aa2-chimpanzee-seal`:
```
datasets 
     -- aa2-chimpanzee-seal
         |-- train.json
         |-- valid.json
         |-- test.json
         |-- label.json
         |-- meta.json
     -- Animals_with_Attributes2
         ...
```

### Run weak supervision methods on generated dataset

[2] Run the weak supervision method on the generated `aa2-chimpanzee-seal` dataset:

```python
import logging
from pathlib import Path

import torch

from wrench.dataset import load_image_dataset
from wrench.endmodel import EndClassifierModel
from wrench.labelmodel import Snorkel
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
dataset_path = Path('../datasets/')
image_root = dataset_path / 'Animals_with_Attributes2'
data = 'aa2-chimpanzee-seal'
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
