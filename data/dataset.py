import datasets as ds
import os
import random
import numpy as np
import json

from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Optional, List

DATASET2TASK_TYPE = {
    "dolly-15k": "causal_lm",
    "alpaca": "causal_lm",
    "natural-instruct": "causal_lm",
    "squad": "causal_lm",
    "ms-marco": "causal_lm",
    "snli": "cls_lm",
    "sst-5": "cls_lm",
    "rte": "cls_lm",
    "wic": "cls_lm",
    "mcl-wic": "cls_lm",
    "race": "cls_lm",
    "boolq": "cls_lm",
    "ag-news": "cls_lm",
    "masakha-news": "cls_lm",
    "yelp": "cls_lm"
}


def load_dataset(
    dataset_name: str,
    dataset_path: str = None,
    down_sample_rate: float = 1.0,
    ratio_train_to_aux: str = None,
    ratio_eval: float = None
) -> Tuple[ds.Dataset, Optional[List[float]], Optional[LabelEncoder]]:
    if dataset_path is None:
        dataset_path = os.path.expanduser("~/.dataset")

    def get_ratios(train_to_aux: str, eval: float, test: float) -> List[float]:
        rs = [float(r) for r in train_to_aux.split(",")]
        rs.append(sum(rs[:2]) * eval / (1 - eval - test))  # add eval ratio
        rs.append(sum(rs[:2]) * test / (1 - eval - test))  # add test ratio
        # normalize
        r_sum = sum(rs)
        rs = [r / r_sum for r in rs]
        return rs

    if dataset_name.lower() == 'dolly-15k':
        dataset_id = "databricks/databricks-dolly-15k"
        dataset = ds.load_dataset(dataset_id, split="train")

        # down sample
        if down_sample_rate < 1.0:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.select(indices[:int(len(dataset) * down_sample_rate)])

        # compute split ratios
        ratios = None
        if ratio_train_to_aux is not None and ratio_eval is not None:
            ratio_test = 0.1
            ratios = get_ratios(ratio_train_to_aux, ratio_eval, ratio_test)

        # label encoding
        label_encoder = LabelEncoder()
        label_encoder.fit(dataset["category"])
        dataset = dataset.map(
            lambda batch: {"class": label_encoder.transform(batch["category"]).tolist()},
            batched=True
        )

        return dataset, ratios, label_encoder
    elif dataset_name.lower() == 'alpaca':
        dataset_id = "tatsu-lab/alpaca"
        dataset = ds.load_dataset(dataset_id, split="train")
        dataset = dataset.rename_column("input", "context")
        dataset = dataset.rename_column("output", "response")

        # down sample
        if down_sample_rate < 1.0:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.select(indices[:int(len(dataset) * down_sample_rate)])

        # compute split ratios
        ratios = None
        if ratio_train_to_aux is not None and ratio_eval is not None:
            ratio_test = 0.1
            ratios = get_ratios(ratio_train_to_aux, ratio_eval, ratio_test)

        return dataset, ratios, None
    elif dataset_name.lower() == 'natural-instruct':
        data_root = os.path.join(dataset_path, 'natural-instructions-2.8')
        dataset = _get_natural_instruct_dataset(data_root, train=True, down_sample_rate=down_sample_rate)

        # compute split ratios
        ratios = None
        if ratio_train_to_aux is not None and ratio_eval is not None:
            ratio_test = 0.1
            ratios = get_ratios(ratio_train_to_aux, ratio_eval, ratio_test)

        # label encoding
        label_encoder = LabelEncoder()
        label_encoder.fit(dataset["category"])
        dataset = dataset.map(
            lambda batch: {"class": label_encoder.transform(batch["category"]).tolist()},
            batched=True
        )

        return dataset, ratios, label_encoder
    elif dataset_name.lower() in ('rte', 'wic', 'boolq'):
        task = dataset_name.lower()
        dataset = ds.load_dataset("super_glue", task, trust_remote_code=True)
        ratio_test = len(dataset["validation"]) / (len(dataset["train"]) + len(dataset["validation"]))
        dataset = dataset.rename_column("label", "class")
        dataset = ds.concatenate_datasets([dataset["train"], dataset["validation"]])

        # down sample
        if down_sample_rate < 1.0:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.select(indices[:int(len(dataset) * down_sample_rate)])

        # compute split ratios
        ratios = None
        if ratio_train_to_aux is not None and ratio_eval is not None:
            ratios = get_ratios(ratio_train_to_aux, ratio_eval, ratio_test)

        return dataset, ratios, None
    elif dataset_name.lower() == 'mcl-wic':
        data_root = os.path.join(dataset_path, 'MCL-WiC')
        train_set, eval_set = _get_mcl_wic_dataset(data_root)
        ratio_test = len(eval_set) / (len(train_set) + len(eval_set))
        dataset = ds.concatenate_datasets([train_set, eval_set])

        # down sample
        if down_sample_rate < 1.0:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.select(indices[:int(len(dataset) * down_sample_rate)])

        # compute split ratios
        ratios = None
        if ratio_train_to_aux is not None and ratio_eval is not None:
            ratios = get_ratios(ratio_train_to_aux, ratio_eval, ratio_test)

        return dataset, ratios, None
    elif dataset_name.lower() == 'race':
        dataset = ds.load_dataset("ehovy/race", "all")
        total_samples = len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"])
        ratio_eval = len(dataset["validation"]) / total_samples
        ratio_test = len(dataset["test"]) / total_samples
        answer_to_class = {"A": 0, "B": 1, "C": 2, "D": 3}
        dataset = dataset.map(
            lambda batch: {"class": [answer_to_class[answer] for answer in batch["answer"]]},
            batched=True
        )
        dataset = ds.concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])

        # down sample
        if down_sample_rate < 1.0:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.select(indices[:int(len(dataset) * down_sample_rate)])

        # compute split ratios
        ratios = None
        if ratio_train_to_aux is not None:
            ratios = get_ratios(ratio_train_to_aux, ratio_eval, ratio_test)

        return dataset, ratios, None
    elif dataset_name.lower() == 'snli':
        dataset = ds.load_dataset("stanfordnlp/snli")
        total_samples = len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"])
        ratio_eval = len(dataset["validation"]) / total_samples
        ratio_test = len(dataset["test"]) / total_samples
        dataset = dataset.rename_column("label", "class")
        dataset = ds.concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])

        # down sample
        if down_sample_rate < 1.0:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.select(indices[:int(len(dataset) * down_sample_rate)])

        # compute split ratios
        ratios = None
        if ratio_train_to_aux is not None:
            ratios = get_ratios(ratio_train_to_aux, ratio_eval, ratio_test)

        return dataset, ratios, None
    elif dataset_name.lower() == 'sst-5':
        dataset = ds.load_dataset("SetFit/sst5")
        total_samples = len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"])
        ratio_eval = len(dataset["validation"]) / total_samples
        ratio_test = len(dataset["test"]) / total_samples
        dataset = dataset.rename_column("label", "class")
        dataset = ds.concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])

        # down sample
        if down_sample_rate < 1.0:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.select(indices[:int(len(dataset) * down_sample_rate)])

        # compute split ratios
        ratios = None
        if ratio_train_to_aux is not None:
            ratios = get_ratios(ratio_train_to_aux, ratio_eval, ratio_test)

        return dataset, ratios, None
    elif dataset_name.lower() == 'squad':
        dataset = ds.load_dataset("rajpurkar/squad")
        ratio_test = len(dataset["validation"]) / (len(dataset["train"]) + len(dataset["validation"]))
        dataset = ds.concatenate_datasets([dataset["train"], dataset["validation"]])

        # down sample
        if down_sample_rate < 1.0:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.select(indices[:int(len(dataset) * down_sample_rate)])

        # label encoding
        label_encoder = LabelEncoder()
        label_encoder.fit(dataset["title"])
        dataset = dataset.map(lambda batch: {"class": label_encoder.transform(batch["title"])}, batched=True)

        # compute split ratios
        ratios = None
        if ratio_train_to_aux is not None and ratio_eval is not None:
            ratios = get_ratios(ratio_train_to_aux, ratio_eval, ratio_test)

        return dataset, ratios, label_encoder
    elif dataset_name.lower() == 'ms-marco':
        dataset = ds.load_dataset("microsoft/ms_marco", "v1.1")
        total_samples = len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"])
        ratio_eval = len(dataset["validation"]) / total_samples
        ratio_test = len(dataset["test"]) / total_samples
        dataset = ds.concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
        dataset = dataset.filter(lambda example: len(example["answers"]) != 0)

        # down sample
        if down_sample_rate < 1.0:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.select(indices[:int(len(dataset) * down_sample_rate)])

        # label encoding
        label_encoder = LabelEncoder()
        label_encoder.fit(dataset["query_type"])
        dataset = dataset.map(lambda batch: {"class": label_encoder.transform(batch["query_type"])}, batched=True)

        # compute split ratios
        ratios = None
        if ratio_train_to_aux is not None:
            ratios = get_ratios(ratio_train_to_aux, ratio_eval, ratio_test)

        return dataset, ratios, label_encoder
    elif dataset_name.lower() == 'ag-news':
        dataset = ds.load_dataset("fancyzhx/ag_news")
        ratio_test = len(dataset["test"]) / (len(dataset["train"]) + len(dataset["test"]))
        dataset = dataset.rename_column("label", "class")
        dataset = ds.concatenate_datasets([dataset["train"], dataset["test"]])

        # down sample
        if down_sample_rate < 1.0:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.select(indices[:int(len(dataset) * down_sample_rate)])

        # compute split ratios
        ratios = None
        if ratio_train_to_aux is not None and ratio_eval is not None:
            ratios = get_ratios(ratio_train_to_aux, ratio_eval, ratio_test)

        return dataset, ratios, None
    elif dataset_name.lower() == 'masakha-news':
        dataset = ds.load_dataset("masakhane/masakhanews", "eng")
        total_samples = len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"])
        ratio_eval = len(dataset["validation"]) / total_samples
        ratio_test = len(dataset["test"]) / total_samples
        dataset = dataset.rename_column("label", "class")
        dataset = ds.concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
        # only keep classes 0 (business), 3 (politics), 5 (sports), 6 (technology)
        class_map = {0: 0, 3: 1, 5: 2, 6: 3}
        dataset = dataset.filter(lambda example: example["class"] in class_map.keys())
        dataset = dataset.map(
            lambda example: {
                "class": [class_map[c] for c in example["class"]]
            }, batched=True
        )

        # down sample
        if down_sample_rate < 1.0:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.select(indices[:int(len(dataset) * down_sample_rate)])

        # compute split ratios
        ratios = None
        if ratio_train_to_aux is not None:
            ratios = get_ratios(ratio_train_to_aux, ratio_eval, ratio_test)

        return dataset, ratios, None
    elif dataset_name.lower() == 'yelp':
        dataset = ds.load_dataset("Yelp/yelp_review_full")
        ratio_test = len(dataset["test"]) / (len(dataset["train"]) + len(dataset["test"]))
        dataset = dataset.rename_column("label", "class")
        dataset = ds.concatenate_datasets([dataset["train"], dataset["test"]])

        # down sample
        if down_sample_rate < 1.0:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.select(indices[:int(len(dataset) * down_sample_rate)])

        # compute split ratios
        ratios = None
        if ratio_train_to_aux is not None and ratio_eval is not None:
            ratios = get_ratios(ratio_train_to_aux, ratio_eval, ratio_test)

        return dataset, ratios, None
    else:
        raise ValueError(f'the dataset {dataset_name} has not been implemented')


def split_dataset(dataset: ds.Dataset, ratios: List[float]) -> List[Optional[ds.Dataset]]:
    datasets = []
    bounds = np.cumsum([0.0] + ratios)
    for lb, ub in zip(bounds[:-1], bounds[1:]):
        lb = int(len(dataset) * lb)
        ub = int(len(dataset) * ub)
        if lb == ub:
            datasets.append(None)
        else:
            datasets.append(dataset.select(range(lb, ub)))
    return datasets


def _get_mcl_wic_dataset(data_root):
    print("load MCL-WiC dataset")

    train_set = []
    with open(os.path.join(data_root, "training", "training.en-en.data"), 'r') as f:
        train_datas = json.load(f)
    with open(os.path.join(data_root, "training", "training.en-en.gold"), 'r') as f:
        train_labels = json.load(f)
    for data, label in zip(train_datas, train_labels):
        tag_to_class = {"F": 0, "T": 1}
        train_set.append({
            "word": data["lemma"],
            "sentence1": data["sentence1"],
            "sentence2": data["sentence2"],
            "class": tag_to_class[label["tag"]]
        })
    train_set = ds.Dataset.from_list(train_set)

    eval_set = []
    with open(os.path.join(data_root, "dev", "multilingual", "dev.en-en.data"), 'r') as f:
        eval_datas = json.load(f)
    with open(os.path.join(data_root, "dev", "multilingual", "dev.en-en.gold"), 'r') as f:
        eval_labels = json.load(f)
    for data, label in zip(eval_datas, eval_labels):
        tag_to_class = {"F": 0, "T": 1}
        eval_set.append({
            "word": data["lemma"],
            "sentence1": data["sentence1"],
            "sentence2": data["sentence2"],
            "class": tag_to_class[label["tag"]]
        })
    eval_set = ds.Dataset.from_list(eval_set)

    return train_set, eval_set


def _get_natural_instruct_dataset(data_root, train=True, num_tasks=100, down_sample_rate=None):
    print(f"load {'train' if train else 'eval'} sets")

    if down_sample_rate is None:
        down_sample_rate = 0.1 if train else 0.01

    with open(os.path.join(
        data_root, 'splits', 'default',
        'train_tasks.txt' if train else 'test_tasks.txt'
    ), 'r') as reader:
        task_splits = [f'{content.strip()}.json' for content in reader.readlines()]

    task_sizes = []
    for task_split in task_splits:
        with open(os.path.join(data_root, 'tasks', task_split)) as reader:
            raw_data = json.load(reader)
            task_sizes.append(len(raw_data["Instances"]))
    task_splits = [task_splits[i] for i in np.argsort(-np.array(task_sizes))[:num_tasks]]

    data = []
    for task_split in task_splits:
        with open(os.path.join(data_root, 'tasks', task_split)) as reader:
            raw_data = json.load(reader)
            instances = raw_data['Instances']

            # down sample the task instances
            instances = np.random.choice(instances, int(down_sample_rate * len(instances)), replace=False)

            instruct = raw_data['Definition'][0]
            for instance in instances:
                # only take the first output into consideration
                data.append({
                    "instruction": instruct,
                    "context": instance['input'],
                    "response": instance['output'][0],
                    "category": task_split.split('_')[0]
                })

    np.random.shuffle(data)
    return ds.Dataset.from_list(data)
