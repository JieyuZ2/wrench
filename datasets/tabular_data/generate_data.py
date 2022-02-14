import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from pprint import pprint

import numpy as np
import openml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
from tqdm import tqdm

ABSTAIN = -1


def get_rules_text(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules


def get_rules(tree):
    tree_ = tree.tree_

    paths = []
    path = []

    features = [
        i if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            threshold = tree_.threshold[node]
            feature = features[node]
            assert feature != 'undefined!'
            p1, p2 = list(path), list(path)
            p1 += [(feature, np.round(threshold, 3), True)]  # <=
            recurse(tree_.children_left[node], p1, paths)
            p2 += [(feature, np.round(threshold, 3), False)]  # >
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    max_dep = 0
    for i, path in enumerate(paths):
        classes = path[-1][0][0]
        label = np.argmax(classes)
        rule = {
            'rule'    : path[:-1],  # feature id, threshold, True:<= | False:>
            'label'   : label,
            'accuracy': np.round(100.0 * classes[label] / np.sum(classes), 2),
            'support' : path[-1][1],
        }
        rules += [rule]
        l = len(path[:-1])
        if l > max_dep:
            max_dep = l

    idx = []
    max_dep_rules = []
    for i, r in enumerate(rules):
        if len(r['rule']) == max_dep:
            idx.append(i)
            max_dep_rules.append(r)

    return max_dep_rules, idx


def apply_rule(x, rule):
    for (feature_id, thres, compare_flag) in rule['rule']:
        if compare_flag:
            if x[feature_id] > thres:
                return ABSTAIN
        else:
            if x[feature_id] <= thres:
                return ABSTAIN
    return rule['label']


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def dump(o, p):
    with open(p, "w+") as file:
        file.write("{\n")
        for i, key in enumerate(list(o.keys())):
            file.write("\t" + f'"{key}"' + ": ")
            json.dump(o[key], file, cls=NpEncoder)
            if i < len(o) - 1:
                file.write(",\n")
        file.write("\n}")


def main(args):
    np.random.seed(args.seed)

    dataset = openml.datasets.get_dataset(args.data_name)
    X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
    if np.any(categorical_indicator):
        X.loc[:, categorical_indicator] = X.loc[:, categorical_indicator].apply(lambda x: pd.factorize(x)[0])
    X = X.to_numpy()
    N = len(X)
    if args.class_encode is None:
        y = pd.factorize(y)
        labels = y[1].to_numpy()
        y = y[0]
    else:
        y.to_numpy()
        labels = args.class_encode
        label_encode = {c: i for i, c in enumerate(labels)}
        y = np.array([label_encode[i] for i in y.tolist()])

    label2idx = defaultdict(list)
    for i, l in enumerate(y):
        label2idx[l].append(i)
    subset_idx = []
    subset_size = int(N * args.used_data)
    subset_label_size = int(subset_size / len(labels))
    for l, idx in label2idx.items():
        subset_idx.extend(np.random.choice(idx, size=min(subset_label_size, len(idx)), replace=False))
    # subset_idx = np.random.choice(N, size=int(N * args.used_data), replace=False)
    X_sub, y_sub = X[subset_idx], y[subset_idx]

    clf = RandomForestClassifier(n_estimators=args.n_trees, max_depth=args.max_depth, max_features=args.max_features, random_state=args.seed)
    clf.fit(X_sub, y_sub)
    trees = clf.estimators_
    rules = []
    rules_text = []
    n_rule = []

    for single_tree in trees:
        cur_rules, idx = get_rules(single_tree)
        rules.append(cur_rules)
        cur_rules_text = get_rules_text(single_tree, feature_names=attribute_names, class_names=labels)
        rules_text.append([cur_rules_text[i] for i in idx])
        n_rule.append(len(cur_rules))

    for i, n in enumerate(n_rule):
        selected = np.random.choice(n)
        rules[i] = rules[i][selected]
        rules_text[i] = rules_text[i][selected]

    Y = y.tolist()
    L = np.array([[apply_rule(xi, ri) for ri in rules] for xi in X])

    L, unique_rule_idx = np.unique(L, axis=1, return_index=True)

    rules = [rules[i] for i in unique_rule_idx]
    rules_text = [rules_text[i] for i in unique_rule_idx]
    for txt in rules_text:
        print(txt)

    indices = np.random.permutation(N)
    train_cut, valid_cut = int(N * .8), int(N * .9)
    train_idx, valid_idx, test_idx = indices[:train_cut], indices[train_cut:valid_cut], indices[valid_cut:]

    rule_label_cnt = Counter([labels[r['label']] for r in rules])
    stats = {
        '# LF'           : len(rules),
        '# LF each label': rule_label_cnt,
        '# train'        : len(train_idx),
        '# valid'        : len(valid_idx),
        '# test'         : len(test_idx),
        'cover rate'     : np.mean(np.sum(L != -1, axis=1) > 0),
    }
    pprint(stats)

    train_json, valid_json, test_json = {}, {}, {}
    data_dir = Path(args.data_dir) / args.data_name
    os.makedirs(data_dir, exist_ok=True)

    for idx, i in tqdm(enumerate(train_idx)):
        train_json[str(idx)] = {"label": Y[i], "weak_labels": L[i], "data": {"feature": X[i].tolist()}}
    dump(train_json, data_dir / 'train.json')

    for idx, i in tqdm(enumerate(valid_idx)):
        valid_json[str(idx)] = {"label": Y[i], "weak_labels": L[i], "data": {"feature": X[i].tolist()}}
    dump(valid_json, data_dir / 'valid.json')

    for idx, i in tqdm(enumerate(test_idx)):
        test_json[str(idx)] = {"label": Y[i], "weak_labels": L[i], "data": {"feature": X[i].tolist()}}
    dump(test_json, data_dir / 'test.json')

    dump({i: c for i, c in enumerate(labels)}, data_dir / 'label.json')

    json.dump({
        'rules': rules,
        'rules_description': rules_text,
        'config':{
            'n_trees': args.n_trees,
            'max_depth': args.max_depth,
            'max_features': args.max_features,
            'used_data': args.used_data,
        }
    }, open(data_dir / 'rules.json', 'w'), cls=NpEncoder, indent=4)


if __name__ == '__main__':
    # mushroom spambase PhishingWebsites Bioresponse bank-marketing
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='mushroom', help="dataset name")
    parser.add_argument('--data_dir', required=False, default='../', help="folder to save dataset")
    parser.add_argument('--class_encode', required=False, default=['0', '1'], nargs='+', )
    parser.add_argument('--seed', default=42, help="random seed")
    parser.add_argument('--used_data', type=float, default=0.1, help="percent of data used for generating rules")
    parser.add_argument('-n', '--n_trees', type=int, default=20)
    parser.add_argument('-d', '--max_depth', type=int, default=3)
    parser.add_argument('-f', '--max_features', default=2)
    args = parser.parse_args()

    label_encode = {
        'spambase'        : ['0', '1'],
        'mushroom'        : ['e', 'p'],
        'bank-marketing'  : ['1', '2'],
        'PhishingWebsites': ['-1', '1'],
        'Bioresponse'     : ['0', '1'],
    }

    args.class_encode = label_encode.get(args.data_name, None)

    main(args)
