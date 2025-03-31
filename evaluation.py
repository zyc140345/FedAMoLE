import numpy as np
import os
import json
import pickle
from typing import Dict, Any

from rouge import Rouge
from tensorboard.backend.event_processing import event_accumulator
from data.dataset import DATASET2TASK_TYPE
from util import list_dir
from distutils.util import strtobool

_rouge = Rouge()


def rouge_score(hyp_ids, ref_ids, tokenizer) -> float:
    hyps = tokenizer.batch_decode(hyp_ids, skip_special_tokens=True)
    if len(hyps[0]) == 0:
        return 0.0
    refs = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

    try:
        rouge_score = _rouge.get_scores(hyps, refs)[0]['rouge-l']['f']
    except ValueError:
        return 0.0

    return rouge_score


ALGO2METRIC_SUFFIX = {
    'learned_adaptive_training': '_mutual_ensemble_adaptive',
    'mutual': '_mutual_ensemble',
    'fed_avg': '',
    'fed_avg_tune': '',
    'fed_prompt': '',
    'fed_ptuning': '',
    'fdlora': '',
    'fed_moe': '',
    'fed_moe_no_amole_rsea': '',
    'fed_moe_no_rsea': '',
    'fed_moe_no_amole': '',
    'fed_moe_no_shared': '',
    'fed_moe_static': '',
    'fed_moe_no_shared_static': '',
    'fed_moe_static_homo': ''
}


def mta(result: Dict[str, Any], algorithm: str) -> np.ndarray:
    metric_name = 'test_acc' + ALGO2METRIC_SUFFIX[algorithm]
    return np.mean(
        [acc for acc in result[metric_name]],  # (num_clients, rounds)
        axis=0
    )  # (rounds,)


def mtr(result: Dict[str, Any], algorithm: str) -> np.ndarray:
    metric_name = 'test_loss' + ALGO2METRIC_SUFFIX[algorithm]
    return np.mean([acc for acc in result[metric_name]], axis=0) * 100


def parse_tensorboard_log(log_dir):
    with open(os.path.join(log_dir, 'params.json'), 'r') as f:
        params = json.load(f)
        client_num = params['client_num']
        task_type = DATASET2TASK_TYPE[params['client_dataset_name']]
    log_file = os.listdir(os.path.join(log_dir, 'summary'))[0]
    log_file = os.path.join(log_dir, 'summary', log_file)
    ea = event_accumulator.EventAccumulator(log_file)
    ea.Reload()
    result = {
        "test_loss": [
            [item.value for item in ea.scalars.Items(f'Loss/Test/Client {i}')]
            for i in range(client_num)
        ]
    }
    if task_type == 'cls_lm':
        result["test_acc"] = [
            [item.value for item in ea.scalars.Items(f'Acc/Test/Client {i}')]
            for i in range(client_num)
        ]
    return result, params


def parse_json_log(log_dir):
    with open(os.path.join(log_dir, 'params.log'), 'r') as f:
        params = {}
        for line in f.readlines():
            k, v = line.split(" : ")
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    try:
                        v = strtobool(v)
                    except ValueError:
                        v = v.strip()
            params[k] = v
    with open(os.path.join(log_dir, 'summary.json'), 'r') as f:
        result = json.load(f)
    return result, params


def summary(root_dir, last=False, return_all=False, include_ft=False):
    datasets = list_dir(root_dir)
    metrics = {}
    result_all = {dataset: {} for dataset in datasets}

    for dataset in datasets:
        print(f"================{dataset} {'last' if last else 'best'}================")
        task_type = DATASET2TASK_TYPE[dataset]
        iids = list_dir(os.path.join(root_dir, dataset))
        datas_var_iids = {}  # <algorithm, (num_iids,)>
        for iid in iids:
            algorithms = list_dir(os.path.join(root_dir, dataset, iid))
            datas = {}
            for algorithm in algorithms:
                seeds = list_dir(os.path.join(root_dir, dataset, iid, algorithm))
                for seed in seeds:
                    timestamp = list_dir(os.path.join(root_dir, dataset, iid, algorithm, seed))[-1]
                    log_dir = os.path.join(root_dir, dataset, iid, algorithm, seed, timestamp)

                    if 'fed_moe' in algorithm:
                        result, params = parse_tensorboard_log(log_dir)
                    else:
                        result, params = parse_json_log(log_dir)

                    if task_type == 'cls_lm':
                        data = mta(result, algorithm)
                    else:
                        data = mtr(result, algorithm)

                    rounds = params["rounds"]
                    if data.shape[0] == 2 * rounds:  # do_ft=True
                        if algorithm in datas:
                            datas[algorithm] = np.vstack([datas[algorithm], data[range(1, data.shape[0], 2)]])
                            if include_ft:
                                datas[algorithm + '_tune'] = np.vstack([
                                    datas[algorithm + '_tune'],
                                    data[range(0, data.shape[0], 2)]
                                ])
                        else:
                            datas[algorithm] = data[range(1, data.shape[0], 2)]
                            if include_ft:
                                datas[algorithm + '_tune'] = data[range(0, data.shape[0], 2)]
                    else:
                        if algorithm in datas:
                            datas[algorithm] = np.vstack([datas[algorithm], data])
                        else:
                            datas[algorithm] = data
                if len(datas[algorithm].shape) == 1:
                    datas[algorithm] = datas[algorithm][np.newaxis, :]
                if algorithm + '_tune' in datas and len(datas[algorithm + '_tune'].shape) == 1:
                    datas[algorithm + '_tune'] = datas[algorithm + '_tune'][np.newaxis, :]

            result_str = f"{iid}: "
            mean_datas = {
                algorithm: data[:, -1].mean() if last else data.max(axis=1).mean()
                for algorithm, data in datas.items()
            }
            for algorithm, data in mean_datas.items():
                if algorithm not in metrics:
                    metrics[algorithm] = []
                metrics[algorithm].append(data)
            std_datas = {
                algorithm: data[:, -1].std() if last else data.max(axis=1).std()
                for algorithm, data in datas.items()
            }
            result_str += ", ".join([
                f"{algorithm}={mean_datas[algorithm]:.2f}±{std_datas[algorithm]:.2f}%"
                for algorithm in mean_datas.keys()
            ])

            if 'fed_moe' in mean_datas:
                best_baseline = max(v for k, v in mean_datas.items() if k != 'fed_moe')
                ours = mean_datas['fed_moe']
                gain = (ours - best_baseline) / best_baseline * 100
                result_str += f", gain={gain:.4f}%"

            print(result_str)

            for algorithm, data in datas.items():
                if algorithm not in datas_var_iids:
                    datas_var_iids[algorithm] = [data[:, -1] if last else data.max(axis=1)]
                else:
                    datas_var_iids[algorithm].append(data[:, -1] if last else data.max(axis=1))

            if return_all:
                result_all[dataset][iid] = datas

        result_str = "Average: "
        mean_datas = {
            algorithm: np.vstack(datas).mean(0).mean()
            for algorithm, datas in datas_var_iids.items()
        }
        std_datas = {
            algorithm: np.vstack(datas).mean(0).std()
            for algorithm, datas in datas_var_iids.items()
        }
        result_str += ", ".join([
            f"{algorithm}={mean_datas[algorithm]:.2f}±{std_datas[algorithm]:.2f}%"
            for algorithm in mean_datas.keys()
        ])
        if 'fed_moe' in mean_datas:
            best_baseline = max(v for k, v in mean_datas.items() if k != 'fed_moe')
            ours = mean_datas['fed_moe']
            gain = (ours - best_baseline) / best_baseline * 100
            result_str += f", gain={gain:.4f}%"
        print(result_str)

    result_str = "Total Average: "
    result_str += ", ".join([
        f"{algorithm}={np.mean(data):.2f}%"
        for algorithm, data in metrics.items()
    ])
    if 'fed_moe' in metrics:
        best_baseline = max(np.mean(v) for k, v in metrics.items() if k != 'fed_moe')
        ours = np.mean(metrics['fed_moe'])
        gain = (ours - best_baseline) / best_baseline * 100
        result_str += f", gain={gain:.4f}%"
    print(result_str)

    if return_all:
        return result_all


def evaluate_stability(root_dir, client_id=0):
    dataset = list_dir(root_dir)[0]
    iid = list_dir(os.path.join(root_dir, dataset))[0]
    algorithm = list_dir(os.path.join(root_dir, dataset, iid))[0]
    seeds = list_dir(os.path.join(root_dir, dataset, iid, algorithm))
    root_dir = os.path.join(root_dir, dataset, iid, algorithm)

    client_dispatch = {}  # <module_name, (rounds, num_experts)>
    client_expert_num = {}  # <module_name, (num_seeds, rounds)>
    for seed_id, seed in enumerate(seeds):
        timestamp = list_dir(os.path.join(root_dir, seed))[0]
        log_dir = os.path.join(root_dir, seed, timestamp)
        with open(os.path.join(log_dir, 'params.json'), 'r') as f:
            params = json.load(f)
            rounds = params['rounds']

        if client_expert_num:
            for module_name in client_expert_num.keys():
                client_expert_num[module_name].append([])
        for r in range(rounds):
            dispatch_path = os.path.join(log_dir, f'dispatch/round{r}_dispatch.pkl')
            with open(dispatch_path, 'rb') as f:
                expert_dispatch = pickle.load(f)

            if seed_id == 0:
                if not client_dispatch:
                    client_dispatch = {
                        module_name: dispatch[:-1, client_id]
                        for module_name, dispatch in expert_dispatch.items()
                    }
                else:
                    for module_name, dispatch in expert_dispatch.items():
                        client_dispatch[module_name] = np.vstack([
                            client_dispatch[module_name], dispatch[:-1, client_id]
                        ])

            if not client_expert_num:
                client_expert_num = {
                    module_name: [[dispatch[:-1, client_id].sum()]]
                    for module_name, dispatch in expert_dispatch.items()
                }
            else:
                for module_name, dispatch in expert_dispatch.items():
                    client_expert_num[module_name][-1].append(dispatch[:-1, client_id].sum())

    client_dispatch = {k: v.astype(int) for k, v in client_dispatch.items()}
    client_expert_num = {k: np.array(v, dtype=int) for k, v in client_expert_num.items()}

    return client_dispatch, client_expert_num


if __name__ == '__main__':
    evaluate_stability('logs', client_id=4)
