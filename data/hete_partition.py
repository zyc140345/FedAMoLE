import numpy as np
import random


def partition_idx_dir(y: np.ndarray, num_parts: int, alpha: float, num_classes: int):
    """
    Partition the data in a practical non-IID way.
    @param y: labels
    @param num_parts: number of parts
    @param alpha: parameter of Dirichlet distribution
    @param num_classes: number of classes in the data
    @return:
    """
    min_size = 0
    min_require_size = 10
    K = num_classes
    N = y.shape[0]
    idx_parts = []

    while min_size < min_require_size:
        idx_parts = [[] for _ in range(num_parts)]
        for k in range(K):
            idx_k = np.where(y == k)[0]  # indices of class k
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_parts))
            proportions = np.array([  # part size should be smaller than the average for balance
                proportion * (len(idx_part) < N / num_parts)
                for proportion, idx_part in zip(proportions, idx_parts)
            ])
            proportions = proportions / proportions.sum()  # normalize
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]  # compute partition points
            idx_k_parts = np.split(idx_k, proportions)  # split idx_k into num_parts parts
            # append to idx_parts
            for idx_part, idx_k_part in zip(idx_parts, idx_k_parts):
                idx_part.extend(idx_k_part.tolist())
        min_size = min([len(idx_part) for idx_part in idx_parts])

    for idx_part in idx_parts:
        np.random.shuffle(idx_part)

    return idx_parts


def partition_idx_meta(y: np.ndarray, num_parts: int, class_per_part: int, num_classes: int):
    """
    Partition the data in a way that each client hold one task split.
    @param y: labels
    @param num_parts: number of parts
    @param class_per_part: number of classes per part
    @param num_classes: number of classes in the data
    @return:
    """
    task_indices = list(range(num_classes))
    task_indices = task_indices[:num_parts * class_per_part]
    idx_parts = []

    for i in range(num_parts):
        idx_parts.append([])
        for task_idx in task_indices[i * class_per_part: (i + 1) * class_per_part]:
            idx_parts[i].extend(np.where(y == task_idx)[0])

    return idx_parts


def partition_idx_pathological(y: np.ndarray, n_parts: int, class_per_part: int, num_classes: int):
    """
    Partition the dataset in a pathological IID way.
    @param y: labels
    @param n_parts: number of parts
    @param class_per_part: number of classes per part
    @param num_classes: number of classes in the data
    @return:
    """
    K = num_classes
    class_cnt = [0 for _ in range(K)]  # number of parts each class appears
    class_parts = []  # classes each part contains

    # try at most 1000 times to find a legal partition
    for _ in range(1000):
        for i in range(n_parts):
            # the i-th part contains the i-th class
            class_part = [i % K]
            class_cnt[i % K] += 1
            # the rest of the classes are randomly selected
            while len(class_part) < class_per_part:
                c = random.randint(0, K - 1)
                if c not in class_part:
                    class_part.append(c)
                    class_cnt[c] += 1
            class_parts.append(class_part)

        # all classes must appear in at least one part
        if len(np.where(np.array(class_cnt) == 0)[0]) == 0:
            break

        # test failed, reset
        class_cnt = [0 for _ in range(K)]
        class_parts = []

    # failed to find a legal partition
    zero_cnt_classes = np.where(np.array(class_cnt) == 0)[0]
    for zero_cnt_class in zero_cnt_classes:
        part_idx = np.arange(n_parts)
        np.random.shuffle(part_idx)
        for i in part_idx:
            # replace a class that appears in multiple parts with the zero_cnt_class
            # so that both classes are legal
            over_one_cnt_classes = np.where(np.array([class_cnt[c] for c in class_parts[i]]) > 1)[0]
            if len(over_one_cnt_classes) > 0:
                replaced_class = over_one_cnt_classes[0]
                class_cnt[class_parts[i][replaced_class]] -= 1
                class_parts[i].pop(replaced_class)
                class_parts[i].append(zero_cnt_class)
                class_cnt[zero_cnt_class] += 1
                break

    # now we have a legal partition, we can construct the idx_parts
    idx_parts = [[] for _ in range(n_parts)]  # indices of each part
    for k in range(K):
        idx_k = np.where(y == k)[0]
        np.random.shuffle(idx_k)
        idx_k_parts = np.array_split(idx_k, class_cnt[k])  # split idx_k into class_cnt[k] parts
        cur_part = 0
        for class_part, idx_part in zip(class_parts, idx_parts):
            if k in class_part:
                idx_part.extend(idx_k_parts[cur_part])
                cur_part += 1

    return idx_parts
