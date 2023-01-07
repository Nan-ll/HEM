import collections
import os
import pickle

import numpy as np


def get_idx(path):

    entities, relations = set(), set()
    for split in ["train", "valid", "test"]:
        with open(os.path.join(path, split), "r") as lines:
            for line in lines:
                lhs, rel, rhs = line.strip().split("\t")
                entities.add(lhs)
                entities.add(rhs)
                relations.add(rel)
    ent2idx = {x: i for (i, x) in enumerate(sorted(entities))}
    rel2idx = {x: i for (i, x) in enumerate(sorted(relations))}
    return ent2idx, rel2idx


def to_np_array(dataset_file, ent2idx, rel2idx):

    examples = []
    with open(dataset_file, "r") as lines:
        for line in lines:
            lhs, rel, rhs = line.strip().split("\t")
            try:
                examples.append([ent2idx[lhs], rel2idx[rel], ent2idx[rhs]])
            except ValueError:
                continue
    return np.array(examples).astype("int64")


def get_filters(examples, n_relations):
    
    lhs_filters = collections.defaultdict(set)
    rhs_filters = collections.defaultdict(set)
    for lhs, rel, rhs in examples:
        rhs_filters[(lhs, rel)].add(rhs)
        lhs_filters[(rhs, rel + n_relations)].add(lhs)
    lhs_final = {}
    rhs_final = {}
    for k, v in lhs_filters.items():
        lhs_final[k] = sorted(list(v))
    for k, v in rhs_filters.items():
        rhs_final[k] = sorted(list(v))
    return lhs_final, rhs_final


def process_dataset(path):

    ent2idx, rel2idx = get_idx(dataset_path)
    examples = {}
    splits = ["train", "valid", "test"]
    for split in splits:
        dataset_file = os.path.join(path, split)
        examples[split] = to_np_array(dataset_file, ent2idx, rel2idx)
    all_examples = np.concatenate([examples[split] for split in splits], axis=0)
    lhs_skip, rhs_skip = get_filters(all_examples, len(rel2idx))
    filters = {"lhs": lhs_skip, "rhs": rhs_skip}
    return examples, filters


if __name__ == "__main__":
    data_path = os.environ["DATA_PATH"]
    for dataset_name in os.listdir(data_path):
        dataset_path = os.path.join(data_path, dataset_name)
        dataset_examples, dataset_filters = process_dataset(dataset_path)
        for dataset_split in ["train", "valid", "test"]:
            save_path = os.path.join(dataset_path, dataset_split + ".pickle")
            with open(save_path, "wb") as save_file:
                pickle.dump(dataset_examples[dataset_split], save_file)
        with open(os.path.join(dataset_path, "to_skip.pickle"), "wb") as save_file:
            pickle.dump(dataset_filters, save_file)
