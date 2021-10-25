import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import pil_loader
from tqdm import trange

DOMAINS = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
INPUT_SIZE = (84, 84)
NUM_WORKERS = 0


class DomainNetDataset(Dataset):
    def __init__(self, root: str, domain: str, class2data: dict, transform=None):
        self.classes = list(class2data.keys())
        self.n_classes = len(self.classes)
        self.root = root
        self.domain = domain
        data, labels, imgs, np_imgs = [], [], [], []
        for c, ds in class2data.items():
            for d in ds:
                p = root / d
                data.append(p)
                img = pil_loader(p).resize(INPUT_SIZE)
                imgs.append(img)
                np_imgs.append(np.asarray(img, dtype='uint8'))
                labels.append(self.classes.index(c))
        self.labels = labels
        self.data = data

        np_imgs = np.asarray(np_imgs) / 255.0
        self.image_mean = np.mean(np_imgs, axis=(0, 1, 2))
        self.image_std = np.std(np_imgs, axis=(0, 1, 2))
        self.imgs = imgs

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std)
            ])
        else:
            self.transform = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.transform(self.imgs[idx])
        label = self.labels[idx]
        return img, label


class DomainNet:
    def __init__(self, data_root):
        data_root = Path(data_root)
        self.data_root = data_root
        domain2data = {}
        classes = set()
        for domain in DOMAINS:
            class2data = {}
            path = data_root / domain
            for c in os.listdir(path):
                classes.add(c)
                p = path / c
                data_list = []
                for image in p.iterdir():
                    data_list.append(image)
                class2data[c] = data_list
            domain2data[domain] = class2data
        self.domain2data = domain2data

        class2data = {c: [] for c in classes}
        for d, dd in domain2data.items():
            for c, data in dd.items():
                class2data[c].extend(data)
        self.class2data = class2data

    def sample_classes(self, k: int):
        # sample k classes with largest number of data
        class2size = {c: len(d) for c, d in self.class2data.items()}
        k_largest = sorted(class2size.items(), key=lambda kv: -kv[1])[:k]
        return [i[0] for i in k_largest]

    def sample_data_from_domain(self, domain: str, classes: list, n_split: int = 1):
        class2data = {c: self.domain2data[domain][c] for c in classes}
        labeler_class2data = [{c: [] for c in classes} for i in range(n_split)]
        for c, data in class2data.items():
            for i, d in enumerate(np.array_split(data, n_split)):
                labeler_class2data[i][c] = d.tolist()
        datasets = [DomainNetDataset(self.data_root, domain, class2data) for c2d in labeler_class2data]
        return datasets


class WeakLabeler(nn.Module):
    '''
    Class for a weak labeler on the Domain Net dataset
    '''

    def __init__(self, n_classes, model_name, pretrained=False, hidden_size=100):
        '''
        Constructor for an endmodel
        '''

        super(WeakLabeler, self).__init__()

        # use un-initialized resnet so that more distinct representations are learned
        self.model = getattr(torchvision.models, model_name)(pretrained=pretrained)
        for param in self.model.parameters():
            param.requires_grad = not pretrained

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, n_classes)

    def forward(self, images):
        '''
        Method for a foward pass of the endmodel
        '''

        resnet_out = self.model(images)
        return resnet_out


def train(train_data: DomainNetDataset, test_data, model_name, pretrained, n_epochs, lr, batch_size, device, test_batch_size=512):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)

    model = WeakLabeler(train_data.n_classes, model_name, pretrained)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    objective_function = torch.nn.CrossEntropyLoss()
    model.to(device)
    best_model = model.state_dict()
    best_acc = -1
    best_epoch = -1

    with trange(n_epochs, unit="epochs", ncols=100, position=0, leave=True) as pbar:
        for ep in range(n_epochs):

            total_ex = 0
            pos_ex = 0

            for x, y in train_dataloader:
                # zeroing gradient
                optimizer.zero_grad()
                # moving data to GPU/CPU
                inputs = x.to(device)
                labels = y.to(device)

                pos_ex += torch.sum(labels).item()
                total_ex += len(labels)

                outputs = model(inputs)
                loss = objective_function(outputs, labels)
                loss.backward()
                optimizer.step()

            train_acc = evaluate(model, train_dataloader, device)
            acc = evaluate(model, test_dataloader, device)
            pbar.update()
            if acc > best_acc:
                best_model = model.state_dict()
                best_acc = acc
                best_epoch = ep
            pbar.set_postfix(ordered_dict={'train acc': train_acc, 'valid acc': acc, 'best_epoch': best_epoch})

    model.load_state_dict(best_model)
    model.cpu()
    return model, train_data.classes


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()

    predictions = []
    labs = []

    for x, y in dataloader:
        np_labels = y.numpy()

        # moving data to GPU/CPU
        inputs = x.to(device)

        _, preds = torch.max(model(inputs), 1)
        np_preds = torch.Tensor.cpu(preds).numpy()

        predictions.append(np_preds)
        labs.append(np_labels)

    predictions = np.concatenate(predictions)
    labs = np.concatenate(labs)
    acc = np.mean(predictions == labs)
    model.train()
    return acc


@torch.no_grad()
def apply(model, dataset, model_classes, device, batch_size=512):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)
    predictions = []
    probas = []

    model.to(device)
    model.eval()
    for x, y in dataloader:
        # moving data to GPU/CPU
        inputs = x.to(device)
        proba = torch.softmax(model(inputs), 1)

        _, preds = torch.max(proba, 1)

        predictions.extend(preds.tolist())
        probas.extend(proba.tolist())

    model.train()
    target_class = target_dataset.classes
    preds = []
    for i in predictions:
        c = model_classes[i]
        if c in target_class:
            preds.append(target_class.index(c))
        else:
            preds.append(-1)
    return predictions, probas


def train_labelers(datasets, model_name, pretrained, n_epochs, lr, batch_size, device):
    if isinstance(datasets, list):
        n = len(datasets)
        if n > 1:
            labelers = []
            model_classes_l = []
            for i in range(n):
                train_data = datasets[i]
                test_data = ConcatDataset([datasets[j] for j in range(n) if j != i])
                labeler, model_classes = train(train_data, test_data, model_name, pretrained, n_epochs, lr, batch_size, device)
                labelers.append(labeler)
                model_classes_l.append(model_classes)
            return labelers, model_classes_l
        else:
            datasets = datasets[0]

    labeler, model_classes = train(datasets, datasets, model_name, pretrained, n_epochs, lr, batch_size, device)
    labelers = [labeler]
    model_classes_l = [model_classes]
    return labelers, model_classes_l


def dataset_split(n, split=(0.8, 0.1, 0.1)):
    assert sum(split) == 1
    perm = np.random.permutation(n)
    s1, s2 = int(n * split[0]), int(n * (split[0] + split[1]))
    train_idx = perm[:s1]
    valid_idx = perm[s1:s2]
    test_idx = perm[s2:]
    return train_idx, valid_idx, test_idx


def dump(o, p):
    with open(p, "w+") as file:
        file.write("{\n")
        for i, key in enumerate(list(o.keys())):
            file.write("\t" + f'"{key}"' + ": ")
            json.dump(o[key], file)
            if i < len(o) - 1:
                file.write(",\n")
        file.write("\n}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./', type=str)
    parser.add_argument('--save_root', default='../', type=str)

    parser.add_argument('--target_domain', default='real', type=str, choices=DOMAINS)
    parser.add_argument('--n_classes', default=5, type=int)
    parser.add_argument('--n_labeler_per_domain', default=1, type=int)
    parser.add_argument('--model_name', default='resnet18', type=str)

    parser.add_argument('--lr', default=0.005, type=float, help="learning rate")
    parser.add_argument('--n_epochs', default=100, type=int, help="number of training epochs")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--pretrained', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda')

    doaminnet_data = DomainNet(args.data_root)
    sampled_classes = doaminnet_data.sample_classes(args.n_classes)
    print(f'sampled classes: {sampled_classes}')

    target_dataset = doaminnet_data.sample_data_from_domain(args.target_domain, sampled_classes, 1)[0]
    labeler_domains = []
    labeler_classes = []
    labeler_predictions = []
    labeler_probas = []
    for d in DOMAINS:
        if d != args.target_domain:
            sampled_datasets = doaminnet_data.sample_data_from_domain(d, sampled_classes, args.n_labeler_per_domain)
            labelers, model_classes_l = train_labelers(sampled_datasets, args.model_name, args.pretrained, args.n_epochs, args.lr, args.batch_size, device)

            for labeler, model_classes in zip(labelers, model_classes_l):
                predictions, probas = apply(labeler, target_dataset, model_classes, device, batch_size=512)
                labeler_domains.append(d)
                labeler_classes.append(model_classes)
                labeler_predictions.append(predictions)
                labeler_probas.append(probas)

    target_classes = target_dataset.classes

    save_dir = Path(f'{args.save_root}/domainnet-{args.target_domain}')
    os.makedirs(save_dir, exist_ok=True)

    label_meta = {i: c for i, c in enumerate(target_classes)}
    lf_meta = {}
    for i in range(len(labeler_domains)):
        lf_meta[i] = {
            'domain'     : labeler_domains[i],
            'label_space': labeler_classes[i],
        }
    data_meta = {
        'lf_meta'   : lf_meta,
        'input_size': INPUT_SIZE,
    }
    json.dump(label_meta, open(save_dir / 'label.json', 'w'))
    json.dump(data_meta, open(save_dir / 'meta.json', 'w'), indent=4)

    labels = target_dataset.labels
    img_path = target_dataset.data
    weak_labels = list(zip(*labeler_predictions))
    weak_probs = list(zip(*labeler_probas))
    processed_data = {}
    for i in trange(len(target_dataset)):
        processed_data[i] = {
            'label'      : labels[i],
            'weak_labels': list(weak_labels[i]),
            'data'       : {
                'image_path': str(img_path[i]),
                'weak_probs': list(weak_probs[i]),
            },
        }

    train_idx, valid_idx, test_idx = dataset_split(len(processed_data))

    train_data = {i: processed_data[idx] for i, idx in enumerate(train_idx)}
    dump(train_data, save_dir / 'train.json')

    valid_data = {i: processed_data[idx] for i, idx in enumerate(valid_idx)}
    dump(valid_data, save_dir / 'valid.json')

    test_data = {i: processed_data[idx] for i, idx in enumerate(test_idx)}
    dump(test_data, save_dir / 'test.json')
