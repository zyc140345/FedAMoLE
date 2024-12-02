import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from typing import Dict, Union
from enum import Enum
from torch.nn.utils.rnn import pad_sequence
from itertools import chain
from data.prompt import DATASET_TO_TEMPLATE


class DefaultToken(Enum):
    PAD_TOKEN = "[PAD]"
    EOS_TOKEN = "</s>"
    BOS_TOKEN = "<s>"
    UNK_TOKEN = "<unk>"
    IGNORE_INDEX = -100


@dataclass
class CausalLMPreprocessor:
    dataset_name: str
    tokenizer: PreTrainedTokenizer
    train: bool = True

    def __call__(self, batch: Dict[str, Union[int, str]]):
        template = DATASET_TO_TEMPLATE[self.dataset_name](self.tokenizer)
        sources, targets = template.get_prompt(batch)

        model_inputs = self.tokenizer(sources)
        source_lens = [len(s) for s in model_inputs["input_ids"]]
        labels = self.tokenizer(targets)["input_ids"]
        if labels[0][0] == self.tokenizer.bos_token_id:
            labels = [label[1:] for label in labels]

        t5 = "t5" in self.tokenizer.__class__.__name__.lower()
        if self.train and not t5:
            batch_size = len(model_inputs["input_ids"])
            for i in range(batch_size):
                model_inputs["input_ids"][i] = model_inputs["input_ids"][i] + labels[i]
                model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i] + [1] * len(labels[i])
                labels[i] = [DefaultToken.IGNORE_INDEX.value] * source_lens[i] + labels[i]
            model_inputs["labels"] = labels
            model_inputs = {
                k: [seq[:self.tokenizer.model_max_length] for seq in v]
                for k, v in model_inputs.items()
            }
            model_inputs.update({
                k: [seq[:-1] + [self.tokenizer.eos_token_id] for seq in v]
                for k, v in model_inputs.items() if k in ('input_ids', 'labels')
            })
        elif t5:  # for Switch Transformer
            model_inputs["labels"] = [
                [self.tokenizer.pad_token_id] + label[:-1] + [self.tokenizer.eos_token_id]
                for label in labels
            ]
        else:
            model_inputs["labels"] = [
                label[:max(self.tokenizer.model_max_length - source_len, 0)]
                for label, source_len in zip(labels, source_lens)
            ]

        model_inputs["source_lens"] = source_lens
        return model_inputs


@dataclass
class ClsLMPreprocessor:
    dataset_name: str
    tokenizer: PreTrainedTokenizer

    def __call__(self, batch: Dict[str, Union[int, str]]):
        template = DATASET_TO_TEMPLATE[self.dataset_name](self.tokenizer, train=False)
        sources, choices = template.get_prompt(batch)

        model_inputs = self.tokenizer(sources)
        source_lens = [len(s) for s in model_inputs["input_ids"]]
        batch_size = len(model_inputs["input_ids"])
        labels = [
            self.tokenizer([choice] * batch_size)["input_ids"]
            for choice in choices
        ]
        labels = [list(label) for label in zip(*labels)]
        if labels[0][0][0] == self.tokenizer.bos_token_id:
            labels = [[choice[1:] for choice in label] for label in labels]

        if 't5' in self.tokenizer.__class__.__name__.lower():  # for Switch Transformer
            for i in range(batch_size):
                model_inputs["input_ids"][i] = [model_inputs["input_ids"][i] for _ in labels[i]]
                model_inputs["attention_mask"][i] = [[1] * len(input_ids) for input_ids in model_inputs["input_ids"][i]]
                labels[i] = [[self.tokenizer.pad_token_id] + label[:-1] + [self.tokenizer.eos_token_id] for label in labels[i]]
            model_inputs["labels"] = labels
        else:
            for i in range(batch_size):
                model_inputs["input_ids"][i] = [model_inputs["input_ids"][i] + label for label in labels[i]]
                model_inputs["attention_mask"][i] = [[1] * len(input_ids) for input_ids in model_inputs["input_ids"][i]]
                labels[i] = [[DefaultToken.IGNORE_INDEX.value] * source_lens[i] + label for label in labels[i]]
            model_inputs["labels"] = labels
            max_length = self.tokenizer.model_max_length
            model_inputs = {
                k: [[seq[:max_length] for seq in seqs] for seqs in v]
                for k, v in model_inputs.items()
            }
            model_inputs.update({
                k: [[seq[:-1] + [self.tokenizer.eos_token_id] for seq in seqs] for seqs in v]
                for k, v in model_inputs.items() if k in ('input_ids', 'labels')
            })

        model_inputs["source_lens"] = source_lens
        return model_inputs


@dataclass
class CausalLMDataCollator:
    tokenizer: PreTrainedTokenizer
    train: bool = True

    def __call__(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(sample["input_ids"][::-1]) for sample in batch],
            batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).flip(dims=[1])
        attention_mask = pad_sequence(
            [torch.tensor(sample["attention_mask"][::-1]) for sample in batch],
            batch_first=True, padding_value=0
        ).flip(dims=[1])
        labels = pad_sequence(
            [torch.tensor(sample["labels"][::-1]) for sample in batch],
            batch_first=True,
            padding_value=DefaultToken.IGNORE_INDEX.value if self.train else self.tokenizer.pad_token_id
        ).flip(dims=[1])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


@dataclass
class ClsLMDataCollator:
    tokenizer: PreTrainedTokenizer

    def __call__(self, batch):
        num_choices = len(batch[0]["input_ids"])
        input_ids = pad_sequence(
            list(chain.from_iterable(
                [torch.tensor(input_ids[::-1]) for input_ids in model_inputs["input_ids"]]
                for model_inputs in batch
            )),
            batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).view(len(batch), num_choices, -1).flip(dims=[2])
        attention_mask = pad_sequence(
            list(chain.from_iterable(
                [torch.tensor(input_ids[::-1]) for input_ids in model_inputs["attention_mask"]]
                for model_inputs in batch
            )),
            batch_first=True, padding_value=0
        ).view(len(batch), num_choices, -1).flip(dims=[2])
        labels = pad_sequence(
            list(chain.from_iterable(
                [torch.tensor(input_ids[::-1]) for input_ids in model_inputs["labels"]]
                for model_inputs in batch
            )),
            batch_first=True, padding_value=DefaultToken.IGNORE_INDEX.value
        ).view(len(batch), num_choices, -1).flip(dims=[2])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "class": torch.tensor([sample["class"] for sample in batch])
        }
