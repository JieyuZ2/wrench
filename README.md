<h1 style="text-align:center">
<img style="vertical-align:middle" width="500" height="180" src="./images/wrench_logo.png" />
</h1>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python3-1f425f.svg?color=purple)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/JieyuZ2/wrench/commits/main)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![repo size](https://img.shields.io/github/repo-size/JieyuZ2/wrench.svg)](https://github.com/JieyuZ2/wrench/archive/master.zip)
[![Total lines](https://img.shields.io/tokei/lines/github/JieyuZ2/wrench?color=red)](https://github.com/JieyuZ2/wrench)
![visitors](https://visitor-badge.glitch.me/badge?page_id=JieyuZ2/wrench)
![GitHub stars](https://img.shields.io/github/stars/JieyuZ2/wrench.svg?color=green)
![GitHub forks](https://img.shields.io/github/forks/JieyuZ2/wrench?color=9cf)
[![Arxiv](https://img.shields.io/badge/ArXiv-2109.11377-orange.svg)](https://arxiv.org/abs/2109.11377) 


## 🔧 New
**1/25/23**
1. Add [Hyper label model](https://github.com/JieyuZ2/wrench/tree/main/examples/run_hyper_label_model.py), please find more details in our [paper](https://openreview.net/forum?id=aCQt_BrkSjC).


**4/20/22**
1. Add [WS explainer](https://github.com/JieyuZ2/wrench/tree/main/examples/run_explainer.py), please find more details in our [paper](https://openreview.net/forum?id=7CONgGdxsV).

**4/20/22**
1. We have updated the `setup.py` to make installation more flexible.

Please use `pip install ws-benchmark==1.1.2rc0` to install the latest version. We strongly suggest create a new environment to install wrench. We will bring better compatibility in the next stable release.
 If you have any problems with installation, please let us know.

Known incompatibilities:

`tensorflow==2.8.0`, `albumentations==0.1.12`

**3/18/22**
1. Wrench is available on [ws-benchmark](https://pypi.org/project/ws-benchmark/) now, using `pip install ws-benchmark` to qucik install.

**2/13/22** 

1. Add [script](https://github.com/JieyuZ2/wrench/tree/main/datasets/tabular_data) to generate LFs for any tabular dataset as well as 5 new tabular datasets, namely, mushroom, spambase, PhishingWebsites, Bioresponse, and bank-marketing.

**11/04/21** 

1. (beta) Add `parallel_fit` for torch model to support pytorch DistributedDataParallel-[example](https://github.com/JieyuZ2/wrench/blob/main/examples/run_torch_ddp.py)

**10/15/21** 

1. A branch of new methods: WeaSEL, ImplyLoss, ASTRA, MeanTeacher, Meta-Weight-Net, Learning-to-Reweight
2. Support image classification (dataset class / torchvision backbone) as well as DomainNet/Animals-with-Attributes2 datasets (check out the `datasets` folder)

## 🔧 What is it?

**Wrench** is a **benchmark platform** containing diverse weak supervision tasks. It also provides a **common and easy framework** for development and evaluation of your own weak supervision models within the benchmark.

For more information, checkout our publications: 
- [WRENCH: A Comprehensive Benchmark for Weak Supervision](https://arxiv.org/abs/2109.11377) (NeurIPS 2021)
- [A Survey on Programmatic Weak Supervision](https://arxiv.org/pdf/2202.05433.pdf)

If you find this repository helpful, feel free to cite our publication:

```
@inproceedings{
zhang2021wrench,
title={{WRENCH}: A Comprehensive Benchmark for Weak Supervision},
author={Jieyu Zhang and Yue Yu and Yinghao Li and Yujing Wang and Yaming Yang and Mao Yang and Alexander Ratner},
booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2021},
url={https://openreview.net/forum?id=Q9SKS5k8io}
}
```

## 🔧 What is weak supervision?

**Weak Supervision** is a paradigm for automated training data creation without manual annotations. 

For a brief overview, please check out this [blog](https://www.snorkel.org/blog/weak-supervision).

For more context, please check out this [survey](https://arxiv.org/abs/2202.05433).

To track recent advances in weak supervision, please follow this [repo](https://github.com/JieyuZ2/Awesome-Weak-Supervision).

## 🔧 Installation
[1] Install anaconda:
Instructions here: https://www.anaconda.com/download/

[2] Clone the repository:
```
git clone https://github.com/JieyuZ2/wrench.git
cd wrench
```

[3] Create virtual environment:
```
conda env create -f environment.yml
source activate wrench
```
If this not working or you want to use only a subset of modules of Wrench, check out this [wiki page](https://github.com/JieyuZ2/wrench/wiki/Environment-Installation)

## 🔧 Available Datasets

The datasets can be downloaded via [this](https://drive.google.com/drive/folders/1v55IKG2JN9fMtKJWU48B_5_DcPWGnpTq?usp=sharing).

**Note that some datasets may have more training examples than what is reported in README/paper because we include the dev set, whose indices can be found in labeled_id.json if exists.**

A documentation of dataset format and usage can be found in this [wiki-page](https://github.com/JieyuZ2/wrench/wiki/Dataset:-Format-and-Usage)

### classification:
| Name | Task | # class | # LF | # train | # validation | # test | data source | LF source |
|:--------|:---------|:------|:------|:------|:-------------|:-------|:------|:------|
| Census | income classification | 2 | 83 | 10083 | 5561         | 16281  | [link](http://archive.ics.uci.edu/ml/datasets/Census+Income) |[link](https://openreview.net/forum?id=SkeuexBtDr) |
| Youtube | spam classification | 2 | 10 | 1586 | 120          | 250    | [link](https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection) | [link](https://github.com/snorkel-team/snorkel-tutorials/tree/master/spam) |
| SMS | spam classification | 2 | 73 | 4571 | 500          | 500    | [link](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) | [link](https://openreview.net/forum?id=SkeuexBtDr) |
| IMDB | sentiment classification | 2 | 8 | 20000 | 2500         | 2500   | [link](https://dl.acm.org/doi/10.5555/2002472.2002491) | [link](https://arxiv.org/abs/2010.04582) |
| Yelp | sentiment classification | 2 | 8 | 30400 | 3800         | 3800   | [link](https://arxiv.org/abs/1509.01626) | [link](https://arxiv.org/abs/2010.04582) |
| AGNews | topic classification | 4 | 9 | 96000 | 12000        | 12000  | [link](https://arxiv.org/abs/1509.01626) | [link](https://arxiv.org/abs/2010.04582) |
| TREC | question classification | 6 | 68 | 4965 | 500          | 500    | [link](https://aclanthology.org/C02-1150.pdf) | [link](https://openreview.net/forum?id=SkeuexBtDr) |
| Spouse | relation classification | 2 | 9 | 22254 | 2801         | 2701   | [link](http://ceur-ws.org/Vol-1568/paper8.pdf) | [link](https://arxiv.org/abs/1711.10160) |
| SemEval | relation classification | 9 | 164 | 1749 | 178            | 600      | [link](https://aclanthology.org/S10-1006/) | [link](https://arxiv.org/abs/1909.02177) |
| CDR | bio relation classification | 2 | 33 | 8430 | 920          | 4673   | [link](https://pubmed.ncbi.nlm.nih.gov/27651457/) | [link](https://arxiv.org/abs/1711.10160) |
| Chemprot | chemical relation classification | 10 | 26 | 12861 | 1607         | 1607   | [link](https://www.semanticscholar.org/paper/Overview-of-the-BioCreative-VI-chemical-protein-Krallinger-Rabal/eed781f498b563df5a9e8a241c67d63dd1d92ad5) | [link](https://arxiv.org/abs/2010.07835) |
| Commercial | video frame classification | 2 | 4 | 64130 | 9479         | 7496   | [link](https://arxiv.org/pdf/2002.11955.pdf) | [link](https://arxiv.org/abs/2002.11955) |
| Tennis Rally | video frame classification | 2 | 6 | 6959 | 746          | 1098   | [link](https://arxiv.org/pdf/2002.11955.pdf) | [link](https://arxiv.org/abs/2002.11955) |
| Basketball | video frame classification | 2 | 4 | 17970 | 1064         | 1222   | [link](https://arxiv.org/pdf/2002.11955.pdf) | [link](https://arxiv.org/abs/2002.11955) |
| [DomainNet](https://github.com/JieyuZ2/wrench/tree/main/datasets/domainnet) | image classification | - | - | - | -            | -      | [link](https://arxiv.org/pdf/1812.01754.pdf) | [link](http://cs.brown.edu/people/sbach/files/mazzetto-icml21.pdf) |

### sequence tagging:
| Name | # class | # LF | # train | # validation | # test | data source | LF source |
|:--------|:---------|:------|:------|:------|:------|:------|:------|
| CoNLL-03 | 4 | 16 | 14041 | 3250 | 3453 | [link](https://arxiv.org/abs/cs/0306050) | [link](https://arxiv.org/abs/2004.14723) |
| WikiGold | 4 | 16 | 1355 | 169 | 170 | [link](https://dl.acm.org/doi/10.5555/1699765.1699767) | [link](https://arxiv.org/abs/2004.14723) |
| OntoNotes 5.0 | 18 | 17 | 115812 | 5000 | 22897 | [link](https://catalog.ldc.upenn.edu/LDC2013T19) | [link]() |
| BC5CDR | 2 | 9 | 500 | 500 | 500 | [link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4860626/) | [link](https://arxiv.org/abs/2105.12848) |
| NCBI-Disease  | 1 | 5 | 592 | 99 | 99 | [link](https://pubmed.ncbi.nlm.nih.gov/24393765/) | [link](https://arxiv.org/abs/2105.12848) |
| Laptop-Review | 1 | 3 | 2436 | 609 | 800 | [link](https://aclanthology.org/S14-2004/) | [link](https://arxiv.org/abs/2105.12848) |
| MIT-Restaurant | 8 | 16 | 7159 | 500 | 1521 | [link](https://groups.csail.mit.edu/sls/publications/2013/Liu_ASRU_2013.pdf) | [link]() |
| MIT-Movies | 12 | 7 | 9241 | 500 | 2441 | [link](http://people.csail.mit.edu/jrg/2013/liu-icassp13.pdf) | [link]() |


The detailed documentation is coming soon.

## 🔧 Available Models

**If you find any of the implementations is wrong/problematic, don't hesitate to raise issue/pull request, we really appreciate it!**

TODO-list: check [this](https://github.com/JieyuZ2/wrench/wiki/TODO-List) out! 

### classification:
| Model                    | Model Type  | Reference                                            | Link to Wrench                                                                                |
|:-------------------------|:------------|:-----------------------------------------------------|:----------------------------------------------------------------------------------------------|
| Majority Voting          | Label Model | --                                                   | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/labelmodel/majority_voting.py#L44)  |
| Weighted Majority Voting | Label Model | --                                                   | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/labelmodel/majority_voting.py#L14)  |
| Dawid-Skene              | Label Model | [link](https://www.jstor.org/stable/2346806)         | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/labelmodel/dawid_skene.py#L15)      |
| Data Progamming          | Label Model | [link](https://arxiv.org/abs/1605.07723)             | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/labelmodel/generative_model.py#L18) |
| MeTaL                    | Label Model | [link](https://arxiv.org/abs/1810.02840)             | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/labelmodel/snorkel.py#L17)          |
| FlyingSquid              | Label Model | [link](https://arxiv.org/pdf/2002.11955)             | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/labelmodel/flyingsquid.py#L16)      |
 | EBCC                     | Label Model | [link](https://proceedings.mlr.press/v97/li19i.html) | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/labelmodel/ebcc.py#L12)             |
| IBCC                     | Label Model | [link](https://proceedings.mlr.press/v97/li19i.html) | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/labelmodel/ibcc.py#L11)             |
| FABLE                    | Label Model | [link](https://arxiv.org/abs/2210.02724)             | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/labelmodel/fable.py#L331)           |
|Hyper Label Model         | Label Model |  [link](https://openreview.net/forum?id=aCQt_BrkSjC) | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/labelmodel/hyper_label_model.py)           |
| Logistic Regression      | End Model   | --                                                   | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/endmodel/linear_model.py#L52)       |
 | MLP                      | End Model   | --                                                   | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/endmodel/neural_model.py#L21)       |
 | BERT                     | End Model   | [link](https://huggingface.co/models)                | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/endmodel/bert_model.py#L23)         |
 | COSINE                   | End Model   | [link](https://arxiv.org/abs/2010.07835)             | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/endmodel/cosine.py#L68)             |
| ARS2                     | End Model   | [link](https://arxiv.org/abs/2210.03092)             | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/endmodel/ars2.py#L53)               |
 | Denoise                  | Joint Model | [link](https://arxiv.org/abs/2010.04582)             | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/classification/denoise.py#L72)      |
 | WeaSEL                   | Joint Model | [link](https://arxiv.org/abs/2107.02233)             | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/classification/weasel.py#L72)       |
 

### sequence tagging:
| Model | Model Type | Reference | Link to Wrench |
|:--------|:---------|:------|:------|
| Hidden Markov Model | Label Model | [link](https://arxiv.org/abs/2004.14723) | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/seq_labelmodel/hmm.py#L81) |
| Conditional Hidden Markov Model | Label Model | [link](https://arxiv.org/abs/2105.12848) | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/seq_labelmodel/chmm.py#L33) |
| LSTM-CNNs-CRF | End Model | [link](https://arxiv.org/abs/1603.01354) | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/seq_endmodel/lstm_crf_model.py#L86) |
| BERT-CRF | End Model | [link](https://huggingface.co/models) | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/seq_endmodel/bert_crf_model.py#L23) |
| LSTM-ConNet | Joint Model | [link](https://arxiv.org/abs/1910.04289) | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/seqtagging/connet.py#L45) |
| BERT-ConNet | Joint Model | [link](https://arxiv.org/abs/1910.04289) | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/seqtagging/connet.py#L210) |

### classification-to-sequence-tagging wrapper:
Wrench also provides a [`SeqLabelModelWrapper`](https://github.com/JieyuZ2/wrench/blob/main/wrench/seq_labelmodel/seq_wrapper.py#L43) that adaptes label model for classification task to sequence tagging task.

### methods from related domains:

#### Robust Learning methods as end model:

| Model | Model Type | Reference | Link to Wrench |
|:--------|:---------|:------|:------|
| Meta-Weight-Net | End Model | [link](https://arxiv.org/abs/1902.07379) | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/metalearning/meta_weight_net.py#L34) |
| Learning2ReWeight | End Model | [link](https://arxiv.org/abs/1803.09050) | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/metalearning/learn_to_reweight.py#L20) |

#### Semi-Supervised Learning methods as end model:

| Model | Model Type | Reference | Link to Wrench |
|:--------|:---------|:------|:------|
| MeanTeacher | End Model | [link](https://arxiv.org/abs/1703.01780) | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/semisupervisedlearning/meanteacher.py#L61) |

#### Weak Supervision with cleaned labels (Semi-Weak Supervision):

| Model | Model Type | Reference | Link to Wrench |
|:--------|:---------|:------|:------|
| ImplyLoss | Joint Model | [link](https://arxiv.org/abs/2004.06025) | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/classification/implyloss.py#L42) |
| ASTRA | Joint Model | [link](https://www.microsoft.com/en-us/research/publication/self-training-weak-supervision-astra/) | [link](https://github.com/JieyuZ2/wrench/blob/main/wrench/classification/astra.py#L87) |

# 🔧  Quick examples


## 🔧  Label model with parallel grid search for hyper-parameters

```python
import logging
import numpy as np
import pprint

from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.search import grid_search
from wrench import labelmodel 
from wrench.evaluation import AverageMeter

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

#### Load dataset 
dataset_home = '../datasets'
data = 'youtube'
train_data, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=False)


#### Specify the hyper-parameter search space for grid search
search_space = {
    'Snorkel': {
        'lr': np.logspace(-5, -1, num=5, base=10),
        'l2': np.logspace(-5, -1, num=5, base=10),
        'n_epochs': [5, 10, 50, 100, 200],
    }
}

#### Initialize label model
label_model_name = 'Snorkel'
label_model = getattr(labelmodel, label_model_name)

#### Search best hyper-parameters using validation set in parallel
n_trials = 100
n_repeats = 5
target = 'acc'
searched_paras = grid_search(label_model(), dataset_train=train_data, dataset_valid=valid_data,
                             metric=target, direction='auto', search_space=search_space[label_model_name],
                             n_repeats=n_repeats, n_trials=n_trials, parallel=True)

#### Evaluate the label model with searched hyper-parameters and average meter
meter = AverageMeter(names=[target])
for i in range(n_repeats):
    model = label_model(**searched_paras)
    history = model.fit(dataset_train=train_data, dataset_valid=valid_data)
    metric_value = model.test(test_data, target)
    meter.update(target=metric_value)

metrics = meter.get_results()
pprint.pprint(metrics)
```

For detailed guidance of `grid_search`, please check out [this wiki page](https://github.com/JieyuZ2/wrench/wiki/Hyperparameter-Search).


## 🔧  Run a standard supervised learning pipeline

```python
import logging
import torch

from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.endmodel import MLPModel

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

#### Load dataset 
dataset_home = '../datasets'
data = 'youtube'

#### Extract data features using pre-trained BERT model and cache it
extract_fn = 'bert'
model_name = 'bert-base-cased'
train_data, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=True, extract_fn=extract_fn,
                                                 cache_name=extract_fn, model_name=model_name)


#### Train a MLP classifier
device = torch.device('cuda:0')
n_steps = 100000
batch_size = 128
test_batch_size = 1000 
patience = 200
evaluation_step = 50
target='acc'

model = MLPModel(n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size)
history = model.fit(dataset_train=train_data, dataset_valid=valid_data, device=device, metric=target, 
                    patience=patience, evaluation_step=evaluation_step)

#### Evaluate the trained model
metric_value = model.test(test_data, target)
```

## 🔧  Build a two-stage weak supervision pipeline

```python
import logging
import torch

from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.endmodel import MLPModel
from wrench.labelmodel import MajorityVoting

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

#### Load dataset 
dataset_home = '../datasets'
data = 'youtube'

#### Extract data features using pre-trained BERT model and cache it
extract_fn = 'bert'
model_name = 'bert-base-cased'
train_data, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=True, extract_fn=extract_fn,
                                                 cache_name=extract_fn, model_name=model_name)

#### Generate soft training label via a label model
#### The weak labels provided by supervision sources are alreadly encoded in dataset object
label_model = MajorityVoting()
label_model.fit(train_data, valid_data)
soft_label = label_model.predict_proba(train_data)


#### Train a MLP classifier with soft label
device = torch.device('cuda:0')
n_steps = 100000
batch_size = 128
test_batch_size = 1000 
patience = 200
evaluation_step = 50
target='acc'

model = MLPModel(n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size)
history = model.fit(dataset_train=train_data, dataset_valid=valid_data, y_train=soft_label, 
                    device=device, metric=target, patience=patience, evaluation_step=evaluation_step)

#### Evaluate the trained model
metric_value = model.test(test_data, target)

#### We can also train a MLP classifier with hard label
from snorkel.utils import probs_to_preds
hard_label = probs_to_preds(soft_label)
model = MLPModel(n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size)
model.fit(dataset_train=train_data, dataset_valid=valid_data, y_train=hard_label, 
          device=device, metric=target, patience=patience, evaluation_step=evaluation_step)
```

## 🔧  Procedural labeling function generator

```python
import logging
import torch

from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.synthetic import ConditionalIndependentGenerator, NGramLFGenerator
from wrench.labelmodel import FlyingSquid

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


#### Generate synthetic dataset
generator = ConditionalIndependentGenerator(
    n_class=2,
    n_lfs=10,
    alpha=0.75, # mean accuracy
    beta=0.1, # mean propensity
    alpha_radius=0.2, # radius of accuracy
    beta_radius=0.1 # radius of propensity
)
train_data = generator.generate_split('train', 10000)
valid_data = generator.generate_split('valid', 1000)
test_data = generator.generate_split('test', 1000)

#### Evaluate label model on synthetic dataset
label_model = FlyingSquid()
label_model.fit(dataset_train=train_data, dataset_valid=valid_data)
target_value = label_model.test(test_data, metric_fn='auc')

#### Load dataset 
dataset_home = '../datasets'
data = 'youtube'

#### Load real-world dataset
train_data, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=False)

#### Generate procedural labeling functions
generator = NGramLFGenerator(dataset=train_data, min_acc_gain=0.1, min_support=0.01, ngram_range=(1, 2))
applier = generator.generate(mode='correlated', n_lfs=10)
L_test = applier.apply(test_data)
L_train = applier.apply(train_data)


#### Evaluate label model on real-world dataset with semi-synthetic labeling functions
label_model = FlyingSquid()
label_model.fit(dataset_train=L_train, dataset_valid=valid_data)
target_value = label_model.test(L_test, metric_fn='auc')
```

## 🔧  Contact

Contact person: Jieyu Zhang, [jieyuzhang97@gmail.com](mailto:jieyuzhang97@gmail.com)

Don't hesitate to send us an e-mail if you have any question.

We're also open to any collaboration!

## 🔧  Contributing Dataset and Model

We sincerely welcome any contribution to the datasets or models!
