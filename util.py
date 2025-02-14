import torch
import random
import numpy as np
import math
import os
import sys
from typing import Dict
from dataclasses import dataclass

StateDict = Dict[str, torch.Tensor]


def set_seed(seed):
    """
    set random seed
    @param seed:
    @return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed == -1:
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.deterministic = True


def get_device(args):
    return torch.device('cuda:{0}'.format(args.gpu))


def get_elapsed_time(start, end):
    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)
    return elapsed_time


def get_total_size(obj, seen=None):
    """Compute the total size of an object in bytes."""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:  # to avoid double counting from recursive references
        return 0

    seen.add(obj_id)
    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(get_total_size(k, seen) + get_total_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_total_size(i, seen) for i in obj)
    elif torch.is_tensor(obj):
        size += obj.element_size() * obj.nelement()  # for torch tensor
    elif isinstance(obj, np.ndarray):
        size += obj.nbytes

    return size


def list_dir(root_dir):
    def hidden_file(path):
        return path.split(os.sep)[-1].startswith('.')
    return list(filter(lambda x: not hidden_file(x), os.listdir(root_dir)))


@dataclass
class ExponentialLR:
    lr_begin: float
    lr_decay: float
    cur_round: int = 0

    def __call__(self) -> float:
        cur_lr = self.lr_begin * math.pow(self.lr_decay, self.cur_round)
        self.cur_round += 1
        return cur_lr


class DescUpdater:
    def __init__(self, progress_bar, task_type: str, train=True, prefix: str = None):
        self.progress_bar = progress_bar

        if prefix is None:
            self.desc = "Train Epoch {}: "
        else:
            self.desc = prefix + ": "

        if train or task_type == "causal_lm":
            self.desc += "loss={:.4f}"
        elif task_type == "cls_lm":
            self.desc += "loss={:.4f}, acc={:.2f} %"
        else:
            raise ValueError(f"Unsupported task type {task_type}")

    def update(self, *items):
        desc = self.desc.format(*items)
        self.progress_bar.set_description(desc)
