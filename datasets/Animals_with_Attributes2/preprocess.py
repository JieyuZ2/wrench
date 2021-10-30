import argparse
import os
import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import trange, tqdm

NUM_WORKERS = 4

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class PreloadImageFolder(ImageFolder):
    def __init__(
            self,
            classes,
            class_to_attribute,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(PreloadImageFolder, self).__init__(root, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        target_to_attribute = [class_to_attribute[c] for c in classes]

        samples, paths = [], []
        for path, target in tqdm(self.samples, desc='loading images'):
            c = self.classes[target]
            if c in classes:
                samples.append((self.loader(path), class_to_idx[c]))
                paths.append(path)
        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in samples]
        self.target_to_attribute = target_to_attribute
        self.attribute_idx = None
        self.paths = paths

    def set_attribute_idx(self, v):
        self.attribute_idx = v

    def __getitem__(self, index):
        sample, target = self.samples[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.attribute_idx is None:
            return sample, self.target_to_attribute[target]
        else:
            return sample, self.target_to_attribute[target][self.attribute_idx]


def read_file(f):
    with open(f) as fin:
        l = [x.strip() for x in fin.readlines()]
    return l


def read_indexed_file(f):
    class_to_index = {}
    with open(f) as fin:
        index = 0
        for line in fin:
            class_name = line.split('\t')[1].strip()
            class_to_index[class_name] = index
            index += 1
    return class_to_index


class AA2:
    def __init__(self, data_root):
        data_root = Path(data_root)
        self.data_root = data_root

        root = Path(data_root)
        classes_path = root / "classes.txt"
        test_path = root / "testclasses.txt"
        train_path = root / "trainclasses.txt"
        images_path = root / "JPEGImages"
        attributes_path = root / "predicates.txt"
        attributes_matrix_path = root / "predicate-matrix-binary.txt"

        self.class_to_index = read_indexed_file(classes_path)
        self.attr_to_index = read_indexed_file(attributes_path)
        self.testclasses = read_file(test_path)
        self.trainclasses = read_file(train_path)
        attributes_matrix = np.array(np.genfromtxt(attributes_matrix_path, dtype='int'))
        self.class_to_attribute = {c: attributes_matrix[idx] for c, idx in self.class_to_index.items()}

        self.train_dataset = PreloadImageFolder(self.trainclasses, self.class_to_attribute, root=images_path, transform=transform)
        self.test_dataset = PreloadImageFolder(self.testclasses, self.class_to_attribute, root=images_path, transform=transform)

    def train_labelers(self, model_name, pretrained, n_epochs, lr, batch_size, device):
        attr_probas = []
        cnt = 0
        test_dataloader = DataLoader(self.test_dataset, batch_size=1024, shuffle=False)
        for a, aid in self.attr_to_index.items():
            labeler = train(self.train_dataset, aid, model_name, pretrained, n_epochs, lr, batch_size, device)
            attr_proba = apply(labeler, test_dataloader, device)
            attr_probas.append(attr_proba)
            cnt += 1
        attr_probas = np.array(attr_probas).T
        return attr_probas


class AttributeDetector(nn.Module):
    def __init__(self, model_name, pretrained=True):
        '''
        Constructor for an endmodel
        '''

        super(AttributeDetector, self).__init__()

        # use un-initialized resnet so that more distinct representations are learned
        self.model = getattr(torchvision.models, model_name)(pretrained=pretrained)
        for param in self.model.parameters():
            param.requires_grad = not pretrained

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, images):
        '''
        Method for a foward pass of the endmodel
        '''

        resnet_out = self.model(images)
        return resnet_out


def train(dataset: PreloadImageFolder, attribute_idx, model_name, pretrained, n_epochs, lr, batch_size, device, test_batch_size=1024):
    dataset.set_attribute_idx(attribute_idx)

    n = len(dataset)
    n_train = int(0.9 * n)
    n_test = n - n_train
    train_data, test_data = random_split(dataset, lengths=[n_train, n_test])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS, persistent_workers=False)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS, persistent_workers=False)

    model = AttributeDetector(model_name, pretrained)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    objective_function = torch.nn.CrossEntropyLoss()
    model.to(device)
    best_model = model.state_dict()
    best_acc = -1
    best_epoch = -1

    with trange(n_epochs * len(train_dataloader), desc=f'training {attribute_idx}-th attribute detector...', unit="batches", ncols=150, position=0, leave=True) as pbar:
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
                pbar.update()

            # train_acc = evaluate(model, train_dataloader, device)
            acc = evaluate(model, test_dataloader, device)
            if acc > best_acc:
                best_model = model.state_dict()
                best_acc = acc
                best_epoch = ep
            # pbar.set_postfix(ordered_dict={'train acc': train_acc, 'valid acc': acc, 'best_epoch': best_epoch})
            pbar.set_postfix(ordered_dict={'valid acc': acc, 'best_epoch': best_epoch})

    model.load_state_dict(best_model)
    model.cpu()
    return model


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
def apply(model, dataloader, device):
    probas = []

    model.to(device)
    model.eval()
    for x, y in tqdm(dataloader, desc='applying weak labeler...'):
        # moving data to GPU/CPU
        inputs = x.to(device)
        proba = torch.softmax(model(inputs), 1)
        probas.extend(proba[:, 1].tolist())
    model.train()
    return probas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./', type=str)

    parser.add_argument('--model_name', default='resnet18', type=str)
    parser.add_argument('--lr', default=0.005, type=float, help="learning rate")
    parser.add_argument('--n_epochs', default=5, type=int, help="number of training epochs")
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--pretrained', action='store_false')
    args = parser.parse_args()

    device = torch.device('cuda:0')

    print(f'Initializing AA2 data...')
    aa2_dataset = AA2(args.data_root)

    save_dir = Path(args.data_root)
    os.makedirs(save_dir, exist_ok=True)

    target_dataset = aa2_dataset.test_dataset
    idx_to_class = target_dataset.classes
    class_id_to_attribute = np.array(target_dataset.target_to_attribute)
    labels = target_dataset.targets
    img_paths = target_dataset.paths

    print(f'Train and apply attribute detector...')
    attr_probas = aa2_dataset.train_labelers(args.model_name, args.pretrained, args.n_epochs, args.lr, args.batch_size, device)

    pickle.dump({
        'classes'           : idx_to_class,
        'class_to_attribute': class_id_to_attribute,
        'labels'            : labels,
        'img_paths'         : img_paths,
        'attr_probas'       : attr_probas,
    }, open(save_dir / 'preprocessed_aa2.pkl', 'wb'))
