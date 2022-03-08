import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union
import torchvision
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets.folder import pil_loader
from PIL import Image
from .basedataset import BaseDataset


def check_weak_labels(dataset: Union[BaseDataset, np.ndarray]) -> np.ndarray:
    if isinstance(dataset, BaseDataset):
        assert dataset.weak_labels is not None, f'Input dataset has no weak labels!'
        L = np.array(dataset.weak_labels)
    else:
        L = dataset
    return L


def split_labeled_unlabeled(dataset: BaseDataset, cut_tied: Optional[bool] = False) -> Tuple[BaseDataset, BaseDataset]:
    weak_labels = np.array(dataset.weak_labels)
    labeled_idx = []
    for i, l in enumerate(weak_labels):
        counter = Counter()
        weaklabels_cnt = 0
        for l_ in l:
            if l_ >= 0:
                counter[l_] += 1
                weaklabels_cnt += 1
        lst = counter.most_common()  # .values()
        if np.max(l) > -1:
            if not cut_tied or (len(lst) > 1 and lst[0][-1] - lst[1][-1] > 0):  # or weaklabels_cnt<=1:
                labeled_idx.append(i)  # = []

    if len(labeled_idx) == len(weak_labels):
        warnings.warn('No unlabeled data found! Use full dataset us unlabeled dataset')
        return dataset, dataset
    else:
        return dataset.create_split(labeled_idx)


def split_conf_unconf_by_percentile(dataset: BaseDataset, y: Optional[np.ndarray] = None, percentile: float = 0.2,
                                    return_y=True, return_thres=False):
    assert percentile > 0 and percentile < 1, f'percentile should be in (0, 1), now it\'s {percentile}!'
    assert len(y) == len(dataset)

    path = dataset.path
    split = dataset.split
    conf_dataset = type(dataset)(path=path)
    conf_dataset.split = split
    unconf_dataset = type(dataset)(path=path)
    unconf_dataset.split = split

    ids = np.array(dataset.ids)
    labels = np.array(dataset.labels)
    examples = np.array(dataset.examples)
    weak_labels = np.array(dataset.weak_labels)
    features = dataset.features

    max_probs = np.max(y, axis=1)

    n = int(len(max_probs) * percentile)
    sort_idx = np.argsort(-max_probs)
    thres = max_probs[sort_idx[n]]

    split_point = n
    while split_point < len(max_probs):
        if max_probs[sort_idx[split_point]] < thres:
            break
        else:
            split_point += 1

    conf_idx = sort_idx[:split_point]
    unconf_idx = sort_idx[split_point:]

    unconf_dataset.ids = ids[unconf_idx]
    unconf_dataset.labels = labels[unconf_idx]
    unconf_dataset.weak_labels = weak_labels[unconf_idx]
    unconf_dataset.examples = examples[unconf_idx]
    if features is not None:
        unconf_dataset.features = features[unconf_idx]

    conf_dataset.ids = ids[conf_idx]
    conf_dataset.labels = labels[conf_idx]
    conf_dataset.weak_labels = weak_labels[conf_idx]
    conf_dataset.examples = examples[conf_idx]
    if features is not None:
        conf_dataset.features = features[conf_idx]

    if return_thres and return_y:
        return (conf_dataset, y[conf_idx]), (unconf_dataset, y[unconf_idx]), (max_probs.max(), max_probs.min(), thres)
    elif return_thres:
        return conf_dataset, unconf_dataset, (max_probs.max(), max_probs.min(), thres)
    elif return_y:
        return (conf_dataset, y[conf_idx]), (unconf_dataset, y[unconf_idx])
    else:
        return conf_dataset, unconf_dataset


def split_conf_unconf(dataset: BaseDataset, y: Optional[np.ndarray] = None, mode: Optional[str] = 'thres',
                      theta: float = 0.2, return_y=True, return_thres=False):
    assert theta > 0 and theta < 1, f'theta should be in (0, 1), now it\'s {theta}!'
    assert len(y) == len(dataset)

    path = dataset.path
    split = dataset.split
    conf_dataset = type(dataset)(path=path)
    conf_dataset.split = split
    unconf_dataset = type(dataset)(path=path)
    unconf_dataset.split = split

    ids = np.array(dataset.ids)
    labels = np.array(dataset.labels)
    examples = np.array(dataset.examples)
    weak_labels = np.array(dataset.weak_labels)
    features = dataset.features

    max_probs = np.max(y, axis=1)

    if mode == 'thres':
        thres = theta
        conf_idx = np.where(max_probs >= thres)[0]
        unconf_idx = np.where(max_probs < thres)[0]
    elif mode == 'percentile':
        n = int(len(max_probs) * theta)
        split_point = n
        sort_idx = np.argsort(-max_probs)
        while split_point < len(max_probs):
            thres = max_probs[sort_idx[split_point]]
            if max_probs[sort_idx[split_point - 1]] == thres:
                split_point += 1
            else:
                break
        conf_idx = sort_idx[:split_point]
        unconf_idx = sort_idx[split_point:]
    else:
        raise NotImplementedError

    unconf_dataset.ids = ids[unconf_idx]
    unconf_dataset.labels = labels[unconf_idx]
    unconf_dataset.weak_labels = weak_labels[unconf_idx]
    unconf_dataset.examples = examples[unconf_idx]
    if features is not None:
        unconf_dataset.features = features[unconf_idx]

    conf_dataset.ids = ids[conf_idx]
    conf_dataset.labels = labels[conf_idx]
    conf_dataset.weak_labels = weak_labels[conf_idx]
    conf_dataset.examples = examples[conf_idx]
    if features is not None:
        conf_dataset.features = features[conf_idx]

    if return_thres and return_y:
        return (conf_dataset, y[conf_idx]), (unconf_dataset, y[unconf_idx]), (max_probs.max(), max_probs.min(), thres)
    elif return_thres:
        return conf_dataset, unconf_dataset, (max_probs.max(), max_probs.min(), thres)
    elif return_y:
        return (conf_dataset, y[conf_idx]), (unconf_dataset, y[unconf_idx])
    else:
        return conf_dataset, unconf_dataset


#### feature extraction for TextDataset
def bag_of_words_extractor(data: List[Dict], **kwargs: Any):
    corpus = list(map(lambda x: x['text'], data))
    count_vect = CountVectorizer(**kwargs)
    corpus_counts = count_vect.fit_transform(corpus).toarray().astype('float32')

    def extractor(data: List[Dict]):
        corpus = list(map(lambda x: x['text'], data))
        return count_vect.transform(corpus).toarray().astype('float32')

    return corpus_counts, extractor


def tf_idf_extractor(data: List[Dict], **kwargs: Any):
    corpus = list(map(lambda x: x['text'], data))
    tfidf_transformer = TfidfVectorizer(**kwargs)
    tfidf = tfidf_transformer.fit_transform(corpus).toarray().astype('float32')

    def extractor(data: List[Dict]):
        corpus = list(map(lambda x: x['text'], data))
        return tfidf_transformer.transform(corpus).toarray().astype('float32')

    return tfidf, extractor


def sentence_transformer_extractor(data: List[Dict], model_name: Optional[str] = 'paraphrase-distilroberta-base-v1', **kwargs: Any):
    corpus = list(map(lambda x: x['text'], data))
    model = SentenceTransformer(model_name, **kwargs)
    embeddings = model.encode(corpus)
    return embeddings, model.encode


def bert_text_extractor(data: List[Dict], device: torch.device = None, model_name: Optional[str] = 'bert-base-cased',
                        feature: Optional[str] = 'cls', **kwargs: Any):
    """
    Extract text features (semantic representation) using pretrain models either by retriving embedding of [CLS] token
    or by averaging all tokens of given sentences.
    :param data: Data in json format.
    :param device: Torch device to be used.
    :param model_name: transformer (Huggingface) model name to be used.
    :param feature: Two options are: "cls" for [CLS], "avr" for average of all tokens.
    :param kwargs: misc arguments for the pretrained model.
    :return: text feature as np array of size (corpus_size, output_dim)
    """

    # assert feature == 'cls' or feature == 'avr', "Please choose from cls and avr as text feature."
    @torch.no_grad()
    def extractor(data: List[Dict]):
        corpus = list(map(lambda x: x['text'], data))
        model = AutoModel.from_pretrained(model_name, **kwargs).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)  # e.g. 'bert-base-cased'
        text_features = []
        for sentence in tqdm(corpus):
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, return_attention_mask=False, return_token_type_ids=False)
            inputs = inputs['input_ids'].to(device)
            if feature == 'cls':
                output = model(inputs).pooler_output.cpu().squeeze().numpy()  # [len(sentence), dim_out]
                text_features.append(output)
            elif feature == 'avr':
                output = model(inputs).last_hidden_state.cpu().squeeze().numpy()  # [len(sentence), dim_out]
                text_features.append(np.average(output, axis=0))
            else:
                raise NotImplementedError
        return np.array(text_features)

    embeddings = extractor(data)
    return embeddings, extractor


#### feature extraction for RelationDataset
def bert_relation_extractor(data: List[Dict], device: torch.device = None,
                            model_name: Optional[str] = 'bert-base-cased',
                            feature: Optional[str] = 'cat', **kwargs: Any):
    """
    R-BERT feature extractor for relation extraction (github.com/monologg/R-BERT).
    :param data: Data in json format.
    :param device: Torch device to be used.
    :param model_name: transformer (Huggingface) model name to be used.
    :param feature: "cat" for concatenated [cls;ent1;ent2]. "avr" for averaged cls, ent1, ent2.
    :param kwargs: Misc arguments for the pretrained model.
    :return: Text feature as np array of size (corpus_size, output_dim * 3 = [cls;ent1;ent2]) for "cat",
            or (corpus_size, output_dim) for "avr",
    """

    # assert feature == 'cat' or feature == 'avr', "Please choose from cat and avr as text feature."

    @torch.no_grad()
    def extractor(data: List[Dict]):
        assert (('span1' in data[0]) and ('span2' in data[0])), "Passed data missing span index."
        assert (('entity1' in data[0]) and ('entity2' in data[0])), "Passed data missing entity name."
        corpus = list(map(lambda x: x['text'], data))
        span1_list, span2_list = list(map(lambda x: x['span1'], data)), list(map(lambda x: x['span2'], data))  # char level
        ent1_list, ent2_list = list(map(lambda x: x['entity1'], data)), list(map(lambda x: x['entity2'], data))
        model = AutoModel.from_pretrained(model_name, **kwargs).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)  # e.g. 'bert-base-cased'
        text_features = []
        for i, sentence in tqdm(enumerate(corpus)):
            span1s, span1n = span1_list[i]  # ent 1 start, ent 1 end
            span2s, span2n = span2_list[i]  # ent 2 start, ent 2 end

            # Assumption - e1s < e2s, e1 and e2 no overlap.
            e1_tkns = tokenizer.tokenize(ent1_list[i])
            e2_tkns = tokenizer.tokenize(ent2_list[i])

            e1_first = span1s < span2s
            if e1_first:
                left_text = sentence[:span1s]
                between_text = sentence[span1n:span2s]
                right_text = sentence[span2n:]
            else:
                left_text = sentence[:span2s]
                between_text = sentence[span2n:span1s]
                right_text = sentence[span1n:]
            left_tkns = tokenizer.tokenize(left_text)
            between_tkns = tokenizer.tokenize(between_text)
            right_tkns = tokenizer.tokenize(right_text)

            if e1_first:
                tokens = ["[CLS]"] + left_tkns + ["$"] + e1_tkns + ["$"] + between_tkns + ["#"] + e2_tkns + [
                    "#"] + right_tkns + ["[SEP]"]
                e1s = len(left_tkns) + 1  # inclusive
                e1n = e1s + len(e1_tkns) + 2  # exclusive
                e2s = e1n + len(between_tkns)
                e2n = e2s + len(e2_tkns) + 2
                end = e2n
            else:
                tokens = ["[CLS]"] + left_tkns + ["#"] + e2_tkns + ["#"] + between_tkns + ["$"] + e1_tkns + [
                    "$"] + right_tkns + ["[SEP]"]
                e2s = len(left_tkns) + 1  # inclusive
                e2n = e2s + len(e2_tkns) + 2  # exclusive
                e1s = e2n + len(between_tkns)
                e1n = e1s + len(e1_tkns) + 2
                end = e1n

            if len(tokens) > 512:
                if end >= 512:
                    len_truncated = len(between_tkns) + len(e1_tkns) + len(e2_tkns) + 6
                    if len_truncated > 512:
                        diff = len_truncated - 512
                        len_between = len(between_tkns)
                        between_tkns = between_tkns[:(len_between - diff) // 2] + between_tkns[(len_between - diff) // 2 + diff:]
                    if e1_first:
                        truncated = ["[CLS]"] + ["$"] + e1_tkns + ["$"] + between_tkns + ["#"] + e2_tkns + ["#"] + [
                            "[SEP]"]
                        e1s = 1  # inclusive
                        e1n = e1s + len(e1_tkns) + 2  # exclusive
                        e2s = e1n + len(between_tkns)
                        e2n = e2s + len(e2_tkns) + 2
                    else:
                        truncated = ["[CLS]"] + ["#"] + e2_tkns + ["#"] + between_tkns + ["$"] + e1_tkns + ["$"] + [
                            "[SEP]"]
                        e2s = 1  # inclusive
                        e2n = e2s + len(e2_tkns) + 2  # exclusive
                        e1s = e2n + len(between_tkns)
                        e1n = e1s + len(e1_tkns) + 2
                    tokens = truncated
                    assert len(tokens) <= 512
                else:
                    tokens = tokens[:512]

            assert e1_tkns == tokens[e1s + 1:e1n - 1]
            assert e2_tkns == tokens[e2s + 1:e2n - 1]
            assert len(tokens) <= 512

            inputs = torch.tensor(tokenizer.convert_tokens_to_ids(tokens), device=device).unsqueeze(0)
            output = model(inputs).last_hidden_state.cpu().squeeze().numpy()  # [len(sentence), dim_out]
            cls_emb = output[0, :]
            ent1_emb, ent2_emb = np.average(output[e1s:e1n, :], axis=0), np.average(output[e2s:e2n, :], axis=0)
            if feature == "cat":
                text_features.append(np.concatenate([cls_emb, ent1_emb, ent2_emb]))
            elif feature == "avr":
                text_features.append(np.average(np.concatenate([cls_emb, ent1_emb, ent2_emb]), axis=0))
            else:
                raise NotImplementedError
        return np.array(text_features)

    embeddings = extractor(data)
    return embeddings, extractor


#### feature extraction for ImageDataset
def image_feature_extractor(data: List[Dict], device: torch.device = None, model_name: Optional[str] = 'resnet18', **kwargs: Any):
    data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    batch_size = 128

    @torch.no_grad()
    def extractor(data: List[Dict]):
        image_paths = list(map(lambda x: x['image_path'], data))
        pretrained_model = getattr(torchvision.models, model_name)(pretrained=True)
        model = nn.Sequential(*list(pretrained_model.children())[:-1])
        model.to(device)
        model.eval()
        image_features = []
        imgs = []
        for ip in tqdm(image_paths):
            imgs.append(data_transform(pil_loader(ip)))

            if len(imgs) == batch_size:
                img_input = torch.stack(imgs).to(device)
                output = model(img_input)
                image_features.append(output.squeeze().cpu().numpy())
                imgs = []

        if len(imgs) > 0:
            img_input = torch.stack(imgs).to(device)
            output = model(img_input)
            image_features.append(output.squeeze().cpu().numpy())

        return np.vstack(image_features)

    embeddings = extractor(data)
    return embeddings, extractor


#### Loading Glove Embedding for Sequence Dataset
def get_glove_embedding(embedding_file_path=None, PAD='PAD', UNK='UNK'):
    f = open(embedding_file_path, 'r', encoding='utf-8').readlines()
    word_dict = {}
    embedding = []
    for i, line in enumerate(f):
        split_line = line.split()
        word = split_line[0]
        embedding.append(np.array([float(val) for val in split_line[1:]]))
        word_dict[word] = i
    embedding = np.array(embedding)

    word_dict[PAD] = len(word_dict)
    word_dict[UNK] = len(word_dict)

    dict_len, embed_size = embedding.shape
    scale = np.sqrt(3.0 / embed_size)
    spec_word = np.random.uniform(-scale, scale, [2, embed_size])
    embedding = np.concatenate([embedding, spec_word], axis=0).astype(np.float)

    return word_dict, embedding
