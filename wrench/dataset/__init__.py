from .basedataset import BaseDataset
from .dataset import NumericDataset, TextDataset, RelationDataset, ImageDataset
from .seqdataset import BaseSeqDataset
from .torchdataset import sample_batch, TorchDataset, BERTTorchTextClassDataset, BERTTorchRelationClassDataset, ImageTorchDataset

numeric_datasets = ['census', 'mushroom', 'spambase', 'PhishingWebsites', 'Bioresponse', 'bank-marketing', 'basketball', 'tennis', 'commercial']
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


def get_dataset_type(dataset_name):
    if dataset_name in numeric_datasets:
        return NumericDataset
    elif dataset_name in text_datasets:
        return TextDataset
    elif dataset_name in relation_dataset:
        return RelationDataset
    elif dataset_name in seq_dataset_list:
        return BaseSeqDataset
    raise NotImplementedError('cannot recognize the dataset type! please specify the dataset_type.')


def load_dataset(data_home, dataset, dataset_type=None, extract_feature=False, extract_fn=None, **kwargs):
    if dataset_type is None:
        dataset_class = get_dataset_type(dataset)
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


def load_image_dataset(data_home, dataset, image_root_path, preload_image=True, extract_feature=False, extract_fn='pretrain', **kwargs):
    dataset_path = Path(data_home) / dataset
    train_data = ImageDataset(path=dataset_path, split='train', image_root_path=image_root_path, preload_image=preload_image)
    valid_data = ImageDataset(path=dataset_path, split='valid', image_root_path=image_root_path, preload_image=preload_image)
    test_data = ImageDataset(path=dataset_path, split='test', image_root_path=image_root_path, preload_image=preload_image)

    if extract_feature:
        extractor_fn = train_data.extract_feature(extract_fn=extract_fn, return_extractor=True, **kwargs)
        valid_data.extract_feature(extract_fn=extractor_fn, return_extractor=False, **kwargs)
        test_data.extract_feature(extract_fn=extractor_fn, return_extractor=False, **kwargs)

    return train_data, valid_data, test_data
