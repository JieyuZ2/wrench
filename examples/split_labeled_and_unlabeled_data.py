import logging

from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

#### Load dataset
dataset_path = f'../datasets/'
data = 'census'
train_data, valid_data, test_data = load_dataset(dataset_path, data, extract_feature=False)

#### Load given labeled data ids
idx, _ = train_data.load_labeled_ids_and_lf_exemplars(f'{dataset_path}/{data}/labeled_ids.json')

#### Sample 100 data as labeled
sampled_idx = train_data.sample(100, return_dataset=False)

#### May have overlap
labeled_ids = list(set(idx + sampled_idx))

#### Split the train dataset into labeled and unlabeled
labeled_data, unlabeled_data = train_data.create_split(labeled_ids)
