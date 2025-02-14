import os
os.environ['TRANSFORMERS_OFFLINE'] = "1"
os.environ['HF_DATASETS_OFFLINE'] = "1"
os.environ['HF_HUB_OFFLINE'] = "1"

import subprocess
import argparse
import time
import itertools
from threading import Thread
from datetime import date
from queue import Queue
from typing import Dict
from distutils.util import strtobool

MOE_ARGS = [
    'expert_num',
    'expert_choices',
    'max_experts',
    'load_balance_alpha',
    'top_k',
    'random_dispatch',
    'save_embs',
    'test_init',
    'static_arch',
    'homo_arch'
]


def run_train_process(
    algorithm: str,
    python_path: str,
    main_file: str,
    max_retry_times: int,
    gpus: Queue,
    kwargs: Dict[str, str]
):
    gpu = gpus.get()
    kwargs['gpu'] = gpu

    print(f"{algorithm} {kwargs['data_he']} {kwargs['seed']} started")
    process_args = [python_path, '-u', main_file]
    for k, v in kwargs.items():
        process_args.extend(['--' + k, str(v)])
    result = subprocess.run(process_args)

    retry_times = 0
    while result.returncode != 0 and retry_times < max_retry_times:
        retry_times += 1
        print(f"{algorithm} {kwargs['data_he']} {kwargs['seed']} failed, retrying {retry_times}/{max_retry_times}")
        result = subprocess.run(process_args)

    if result.returncode == 0:
        print(f"{algorithm} {kwargs['data_he']} {kwargs['seed']} finished")
    else:
        print(f"{algorithm} {kwargs['data_he']} {kwargs['seed']} failed, max retry times reached")

    gpus.put(gpu)


if __name__ == '__main__':
    # ===== Parse arguments =====
    today = date.today()
    parser = argparse.ArgumentParser(description='Federated MoE')

    # FL settings
    parser.add_argument('--client_num', default=10, type=int,
                        help='the number of clients.')
    parser.add_argument('--rounds', default=30, type=int,
                        help='the number of rounds for federated averaging.')
    parser.add_argument('--local_epochs', default=1, type=int,
                        help='the number of local epochs before sharing the local updates.')
    parser.add_argument('--client_step', default=None, type=int,
                        help='the number of steps for each client to train in each round.')
    parser.add_argument('--do_ft', default=False, type=strtobool,
                        help='whether to finetune the global model after aggregation.')

    # Model settings
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--precision', default='fp16', type=str,
                        help='the backbone precision, optional in ["fp16", "fp32"].')
    parser.add_argument('--lora_rank', default=8, type=int,
                        help='the rank of lora adapter.')
    parser.add_argument('--lora_alpha', default=16, type=float,
                        help='the alpha of lora adapter.')
    parser.add_argument('--lora_dropout', default=0.05, type=int,
                        help='the dropout ratio of lora adapter.')

    # MoE settings
    parser.add_argument('--expert_num', default=30, type=int,
                        help='the number of total experts.')
    parser.add_argument('--expert_choices', default=2, type=int,
                        help='the number of clients chosen by each expert.')
    parser.add_argument('--max_experts', default=8, type=int,
                        help='the maximum number of experts for each client.')
    parser.add_argument('--load_balance_alpha', default=0.001, type=float,
                        help='the weight of the load balance loss.')
    parser.add_argument('--top_k', default=2, type=int,
                        help='the number of experts (exclude the shared expert) selected by each token.')
    parser.add_argument('--random_dispatch', default=False, type=strtobool,
                        help='whether to randomly dispatch the experts to the clients.')
    parser.add_argument('--save_embs', default=False, type=strtobool,
                        help='whether to save the embeddings of the experts and the shared expert.')
    parser.add_argument('--test_init', default=False, type=strtobool,
                        help='whether to test test the initial model.')
    parser.add_argument('--static_arch', default=False, type=strtobool,
                        help='Whether to use static model architectures throughout the FL progress.')
    parser.add_argument('--homo_arch', default=False, type=strtobool,
                        help='Whether to use homogeneous model architectures across clients..')

    # Optimizer settings
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='the learning rate of local training.')
    parser.add_argument('--lr_decay', default=0.99, type=float,
                        help='learning rate decay.')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--algorithms', default='fed_avg,fed_moe', type=str,
                        help='optional in ["learned_adaptive_training", "mutual". '
                             '"fed_avg", "fed_prompt", "fed_ptuning", "fed_moe"]')

    # Data settings
    parser.add_argument('--client_dataset_name', type=str, required=True,
                        help='the name of the client dataset.')
    parser.add_argument('--client_dataset_ratio', default=1.0, type=float,
                        help='the ratio of the client dataset to the total dataset.')
    parser.add_argument('--data_hes', type=str, required=True,
                        help='the heterogeneity of the client data distributions, '
                             'optional in ["iid", "dir<alpha>", "meta<class_per_client>"].')
    parser.add_argument('--ratio_train_to_aux', default='0.95,0.05', type=str,
                        help='the ratio of train set to auxiliary set.')
    parser.add_argument('--ratio_eval', default=0.1, type=float,
                        help='the ratio of the eval set to the down-sampled dataset.')
    parser.add_argument('--max_length', default=1024, type=int,
                        help='the max length of sequence.')

    # Other settings
    parser.add_argument('--python_path', default='/home/zyc/miniconda3/bin/python', type=str,
                        help='the path of python interpreter.')
    parser.add_argument('--gpus', default='0,1,2', type=str,
                        help='the indices of GPUs to use.')
    parser.add_argument('--log_level', default='summarized', type=str,
                        help='the level of logging, optional in ["summarized", "detailed"].')
    parser.add_argument('--seeds', default='42,62,82', type=str,
                        help='the random seed for reproducibility.')
    parser.add_argument('--max_retry_times', default=2, type=int)

    args = parser.parse_args()

    algorithms = args.algorithms.split(',')
    data_hes = args.data_hes.split(',')
    seeds = [int(seed) for seed in args.seeds.split(',')]
    gpus = Queue(maxsize=8)
    for gpu in args.gpus.split(','):
        gpus.put(gpu)

    threads = []
    for seed, data_he, algorithm in itertools.product(seeds, data_hes, algorithms):
        time_stamp = int(time.time())
        thread_args = {k: v for k, v in args.__dict__.items() if v is not None}
        del thread_args['python_path']
        del thread_args['max_retry_times']
        del thread_args['data_hes']
        del thread_args['gpus']
        del thread_args['seeds']
        del thread_args['algorithms']
        thread_args['data_he'] = data_he
        thread_args['seed'] = seed
        thread_args['time_stamp'] = time_stamp
        thread_args['log_file'] = True

        if algorithm == 'fed_moe':
            del thread_args['do_ft']
            train_script = os.path.join(os.path.dirname(__file__), '../main.py')
        else:
            thread_args = {k: v for k, v in thread_args.items() if k not in MOE_ARGS}
            thread_args['algorithm'] = algorithm
            train_script = os.path.join(os.path.dirname(__file__), '../train_baselines.py')

        thread = Thread(
            target=run_train_process,
            args=(
                algorithm, args.python_path, train_script,
                args.max_retry_times, gpus, thread_args
            )
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print("All finished")
