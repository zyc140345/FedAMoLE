import numpy as np
from torch.utils.data import DataLoader
from data.dataset import split_dataset
from data.hete_partition import partition_idx_dir, partition_idx_meta
from data.preprocess import CausalLMDataCollator, ClsLMDataCollator, CausalLMPreprocessor, ClsLMPreprocessor
from model.common import get_tokenizer


class ClientDataLoader:
    def __init__(self, dataset, ratios, label_encoder, args):
        self.ratios, self.label_encoder = ratios, label_encoder
        self.tokenizer = get_tokenizer(args)

        if args.task_type == "causal_lm":
            train_collator = CausalLMDataCollator(self.tokenizer)
            test_collator = CausalLMDataCollator(self.tokenizer, train=False)
        else:
            train_collator = CausalLMDataCollator(self.tokenizer)
            test_collator = ClsLMDataCollator(self.tokenizer)

        if 'class' not in dataset[0].keys() or args.data_he == "iid":
            idx = np.arange(len(dataset))
            np.random.shuffle(idx)
            idx_parts = np.array_split(idx, args.client_num)
        else:
            y = np.array(dataset["class"])
            num_classes = np.unique(y).shape[0]
            if 'dir' in args.data_he:
                alpha = float(args.data_he[3:])
                idx_parts = partition_idx_dir(y, args.client_num, alpha=alpha, num_classes=num_classes)
            elif 'meta' in args.data_he:
                class_per_part = int(args.data_he[4:])
                idx_parts = partition_idx_meta(y, args.client_num, class_per_part, num_classes=num_classes)
            else:
                raise ValueError(f'Unknown non-IID setting {args.data_he}')

            for client_id, idx_part in enumerate(idx_parts):
                np.random.shuffle(idx_part)
                if args.log_level == "detailed":
                    distribution = np.bincount(y[idx_part], minlength=num_classes)
                    print(f"Client {client_id} class distribution: {distribution}")

        self.train_sets = []
        self.aux_sets = []
        self.eval_sets = []
        self.test_sets = []

        for client_id in range(args.client_num):
            train_set, aux_set, eval_set, test_set = split_dataset(dataset.select(idx_parts[client_id]), self.ratios)
            if aux_set and len(aux_set) > 200:
                aux_set = aux_set.select(range(200))
            if len(eval_set) > 200:
                eval_set = eval_set.select(range(200))
            if args.task_type == "cls_lm" and len(test_set) > 200:
                test_set = test_set.select(range(200))
            elif args.task_type == "causal_lm" and len(test_set) > 50:
                test_set = test_set.select(range(50))
            self.train_sets.append(train_set)
            self.aux_sets.append(aux_set)
            self.eval_sets.append(eval_set)
            self.test_sets.append(test_set)

        train_preprocessor = None
        test_preprocessor = None
        if args.task_type == "causal_lm":
            train_preprocessor = CausalLMPreprocessor(args.client_dataset_name, self.tokenizer)
            test_preprocessor = CausalLMPreprocessor(args.client_dataset_name, self.tokenizer, train=False)
        elif args.task_type == "cls_lm":
            train_preprocessor = CausalLMPreprocessor(args.client_dataset_name, self.tokenizer)
            test_preprocessor = ClsLMPreprocessor(args.client_dataset_name, self.tokenizer)

        for client_id in range(args.client_num):
            for set_prefix in ("train", "aux", "eval", "test"):
                preprocessor = test_preprocessor if set_prefix == "test" else train_preprocessor
                sets = getattr(self, set_prefix + "_sets")
                if sets[client_id]:
                    sets[client_id] = sets[client_id].map(preprocessor, batched=True, load_from_cache_file=False)
                    max_length = self.tokenizer.model_max_length  # avoid memory leak
                    sets[client_id] = sets[client_id].filter(lambda example: example["source_lens"] < max_length)
                    if args.task_type == "causal_lm":
                        sets[client_id].set_format(columns=["input_ids", "attention_mask", "labels"])
                    else:
                        sets[client_id].set_format(columns=["input_ids", "attention_mask", "labels", "class"])

        self.train_loaders = [
            DataLoader(
                train_set, args.batch_size, shuffle=True, pin_memory=True,
                collate_fn=train_collator
            ) for train_set in self.train_sets
        ]
        self.aux_loaders = [
            DataLoader(
                aux_set, min(args.batch_size, len(aux_set)), shuffle=False, pin_memory=True,
                collate_fn=train_collator
            ) if aux_set else None
            for aux_set in self.aux_sets
        ]
        self.eval_loaders = [
            DataLoader(
                eval_set, args.batch_size, shuffle=False, pin_memory=True,
                collate_fn=train_collator
            ) for eval_set in self.eval_sets
        ]
        self.test_loaders = [
            DataLoader(
                test_set, args.batch_size, shuffle=False, pin_memory=True,
                collate_fn=test_collator
            ) for test_set in self.test_sets
        ]
