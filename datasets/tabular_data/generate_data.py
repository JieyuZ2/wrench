import argparse
import json
import os
from pathlib import Path

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

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [(node, np.round(threshold, 3), True)]  # <=
            recurse(tree_.children_left[node], p1, paths)
            p2 += [(node, np.round(threshold, 3), False)]  # >
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
        classes = path[-1][0][0]
        label = np.argmax(classes)
        rule = {
            'rule'    : path[:-1],  # feature id, threshold, True:<= | False:>
            'label'   : label,
            'accuracy': np.round(100.0 * classes[label] / np.sum(classes), 2),
            'support' : path[-1][1],
        }
        rules += [rule]

    return rules


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
    y = pd.factorize(y)

    clf = RandomForestClassifier(n_estimators=args.n_trees, max_depth=args.max_depth, random_state=args.seed)
    clf.fit(X, y[0])
    trees = clf.estimators_
    rules = []
    rules_text = []

    for single_tree in trees:
        cur_rules = get_rules(single_tree)
        rules.extend(cur_rules)
        cur_rules_text = get_rules_text(single_tree, feature_names=attribute_names, class_names=y[1].to_numpy())
        rules_text.extend(cur_rules_text)

    for txt in rules_text:
        print(txt)

    X = X.to_numpy()
    Y = y[0].tolist()
    L = [[apply_rule(xi, ri) for ri in rules] for xi in X]
    N = len(Y)

    indices = np.random.permutation(N)
    train_cut, valid_cut = int(N * .8), int(N * .9)
    train_idx, valid_idx, test_idx = indices[:train_cut], indices[train_cut:valid_cut], indices[valid_cut:]

    train_json, valid_json, test_json = {}, {}, {}
    data_dir = Path(args.data_dir) / args.data_name
    os.makedirs(data_dir, exist_ok=True)

    for idx, i in tqdm(enumerate(train_idx)):
        train_json[str(idx)] = {"label": Y[i], "weak_labels": L[i], "data": {"feature": X[i].tolist()}}
    dump(train_json, data_dir / 'train.json')

    for idx, i in tqdm(enumerate(valid_idx)):
        train_json[str(idx)] = {"label": Y[i], "weak_labels": L[i], "data": {"feature": X[i].tolist()}}
    dump(train_json, data_dir / 'valid.json')

    for idx, i in tqdm(enumerate(test_idx)):
        train_json[str(idx)] = {"label": Y[i], "weak_labels": L[i], "data": {"feature": X[i].tolist()}}
    dump(train_json, data_dir / 'test.json')

    dump({i: c for i, c in enumerate(y[1].to_numpy())}, data_dir / 'label.json')

    json.dump({'rules': rules, 'rules_description': rules_text}, open(data_dir / 'rules.json', 'w'), cls=NpEncoder, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', required=True, help="dataset name", default='mushroom')
    parser.add_argument('--data_dir', required=False, default='./', help="folder to save dataset")
    parser.add_argument('--seed', default=42, help="random seed")
    parser.add_argument('-n', '--n_trees', type=int, default=10)
    parser.add_argument('-d', '--max_depth', type=int, default=2)
    args = parser.parse_args()

    main(args)
