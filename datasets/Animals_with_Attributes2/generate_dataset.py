import argparse
import json
import pickle
from tqdm import trange
import os
from pathlib import Path
import numpy as np

ATTRIBUTES = ['black', 'white', 'blue', 'brown', 'gray', 'orange', 'red', 'yellow', 'patches', 'spots', 'stripes', 'furry', 'hairless', 'toughskin', 'big', 'small', 'bulbous', 'lean', 'flippers', 'hands', 'hooves', 'pads', 'paws', 'longleg', 'longneck', 'tail', 'chewteeth', 'meatteeth', 'buckteeth', 'strainteeth', 'horns', 'claws', 'tusks', 'smelly', 'flys', 'hops', 'swims', 'tunnels', 'walks', 'fast', 'slow', 'strong', 'weak', 'muscle', 'bipedal', 'quadrapedal', 'active', 'inactive', 'nocturnal', 'hibernate', 'agility', 'fish', 'meat', 'plankton', 'vegetation', 'insects', 'forager', 'grazer', 'hunter', 'scavenger', 'skimmer', 'stalker', 'newworld', 'oldworld', 'arctic', 'coastal', 'desert', 'bush', 'plains', 'forest', 'fields', 'jungle', 'mountains', 'ocean', 'ground', 'water', 'tree', 'cave', 'fierce', 'timid', 'smart', 'group', 'solitary', 'nestspot', 'domestic']

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
    parser.add_argument('--path', default='./preprocessed_aa2.pkl', type=str)
    parser.add_argument('--save_root', default='../', type=str)

    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--unipolar', action='store_true')
    parser.add_argument('--class1', default='chimpanzee', type=str, choices=['chimpanzee', 'giant+panda', 'leopard', 'persian+cat', 'pig', 'hippopotamus', 'humpback+whale', 'raccoon', 'rat', 'seal'])
    parser.add_argument('--class2', default='seal', type=str, choices=['chimpanzee', 'giant+panda', 'leopard', 'persian+cat', 'pig', 'hippopotamus', 'humpback+whale', 'raccoon', 'rat', 'seal'])

    args = parser.parse_args()

    unipolar = args.unipolar

    print(f'load preprocessed data from {args.path}...')
    preprocessed = pickle.load(open(args.path, 'rb'))
    idx_to_class = preprocessed['classes']
    class_id_to_attribute = preprocessed['class_to_attribute']
    labels = np.array(preprocessed['labels'])
    img_paths = np.array(preprocessed['img_paths'])
    attr_probas = preprocessed['attr_probas']

    c1, c2 = args.class1, args.class2
    assert c1 in idx_to_class and c2 in idx_to_class
    print(f'generate binary weak supervision task for {c1}/{c2}...')

    cid1, cid2 = idx_to_class.index(c1), idx_to_class.index(c2)
    data_idx = (labels==cid1) | (labels==cid2)
    labels = labels[data_idx]
    img_paths = img_paths[data_idx]
    attr_probas = attr_probas[data_idx]

    attr1, attr2 = class_id_to_attribute[cid1], class_id_to_attribute[cid2]
    diff_attr = np.logical_xor(attr1, attr2)
    selected_attr = [ATTRIBUTES[i] for i, a in enumerate(diff_attr) if a]

    attr_probas = attr_probas[:, diff_attr]
    attr1, attr2 = attr1[diff_attr], attr2[diff_attr]
    attr_to_class = attr2.copy()

    threshold = args.threshold
    L = np.empty_like(attr_probas)
    for i, c in enumerate(attr_to_class):
        pos_mask = attr_probas[:, i] > threshold
        neg_mask = ~ pos_mask
        L[pos_mask, i] = c
        if unipolar:
            L[neg_mask, i] = -1
        else:
            L[neg_mask, i] = 1 - c
    conf_matrix = attr_probas * attr_to_class + (1-attr_to_class) * (1-attr_probas)

    label_meta = {0: c1, 1:c2}
    c1_mask = labels == cid1
    c2_mask = labels == cid2
    labels[c1_mask] = 0
    labels[c2_mask] = 1

    processed_data = {}
    for i in trange(len(labels)):
        processed_data[i] = {
            'label'      : int(labels[i]),
            'weak_labels': list(L[i]),
            'data'       : {
                'image_path': str(img_paths[i]),
                'weak_probs': [[c, 1-c] for c in conf_matrix[i]],
            },
        }

    save_dir = Path(args.save_root) / f'aa2-{c1}-{c2}'
    os.makedirs(save_dir, exist_ok=True)
    print(f'save dataset to {save_dir}...')

    lf_meta = {k:{'attribute': v} for k,v in enumerate(selected_attr)}
    data_meta = {
        'lf_meta'   : lf_meta,
        'input_size': (224, 224),
    }
    json.dump(label_meta, open(save_dir / 'label.json', 'w'))
    json.dump(data_meta, open(save_dir / 'meta.json', 'w'), indent=4)

    train_idx, valid_idx, test_idx = dataset_split(len(processed_data))

    train_data = {i: processed_data[idx] for i, idx in enumerate(train_idx)}
    dump(train_data, save_dir / 'train.json')

    valid_data = {i: processed_data[idx] for i, idx in enumerate(valid_idx)}
    dump(valid_data, save_dir / 'valid.json')

    test_data = {i: processed_data[idx] for i, idx in enumerate(test_idx)}
    dump(test_data, save_dir / 'test.json')
