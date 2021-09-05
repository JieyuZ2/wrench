from .basedataset import BaseDataset
from .dataset import NumericDataset, TextDataset, RelationDataset, EntityDataset
from .seqdataset import BaseSeqDataset
from .torchdataset import TorchDataset, sample_batch, BERTTorchTextClassDataset, BERTTorchRelationClassDataset

numeric_datasets = ['census', 'basketball', 'tennis', 'commercial']
text_datasets = ['agnews', 'imdb', 'sms', 'trec', 'yelp', 'youtube']
relation_dataset = ['cdr', 'spouse', 'chemprot', 'semeval']
cls_dataset_list = numeric_datasets + text_datasets + relation_dataset
bin_cls_dataset_list = numeric_datasets + ['cdr', 'spouse', 'sms', 'yelp', 'imdb', 'youtube']
seq_dataset_list = ['laptopreview', 'ontonotes', 'ncbi-disease', 'bc5cdr', 'mit-restaurants', 'mit-movies', 'wikigold', 'conll']

import shutil
from pathlib import Path
from os import environ, makedirs
from os.path import expanduser, join


#### dataset downloading and loading
def get_data_home(data_home=None) -> str:
    data_home = data_home or environ.get('WRENCH_DATA', join('~', 'wrench_data'))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def load_dataset(data_home, dataset, dataset_type=None, extract_feature=False, extract_fn=None, **kwargs):
    if dataset_type is None:
        if dataset in numeric_datasets:
            dataset_class = NumericDataset
        elif dataset in text_datasets:
            dataset_class = TextDataset
        elif dataset in relation_dataset:
            dataset_class = RelationDataset
        elif dataset in seq_dataset_list:
            dataset_class = BaseSeqDataset
        else:
            raise NotImplementedError('cannot recognize the dataset type! please specify the dataset_type.')
    else:
        dataset_class = eval(dataset_type)

    dataset_path = Path(data_home) / dataset
    train_data = dataset_class(path=dataset_path, split='train')
    valid_data = dataset_class(path=dataset_path, split='valid')
    test_data = dataset_class(path=dataset_path, split='test')

    if extract_feature and (dataset_class != BaseSeqDataset):
        extractor_fn = train_data.extract_feature(extract_fn=extract_fn, return_extractor=True, **kwargs)
        valid_data.extract_feature(extract_fn=extractor_fn, return_extractor=False, **kwargs)
        test_data.extract_feature(extract_fn=extractor_fn, return_extractor=False, **kwargs)

    return train_data, valid_data, test_data
