import subprocess
import os
import argparse
import time
import itertools
from threading import Thread
from queue import Queue
from typing import Dict
from distutils.util import strtobool


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

    print(f"{algorithm} {kwargs['client_num']} clients started")
    process_args = [python_path, '-u', main_file]
    for k, v in kwargs.items():
        process_args.extend(['--' + k, str(v)])
    result = subprocess.run(process_args)

    retry_times = 0
    while result.returncode != 0 and retry_times < max_retry_times:
        retry_times += 1
        print(f"{algorithm} {kwargs['client_num']} clients failed, retrying {retry_times}/{max_retry_times}")
        result = subprocess.run(process_args)

    if result.returncode == 0:
        print(f"{algorithm} {kwargs['client_num']} clients finished")
    else:
        print(f"{algorithm} {kwargs['client_num']} clients failed, max retry times reached")

    gpus.put(gpu)


if __name__ == '__main__':
    # ===== Parse arguments =====
    parser = argparse.ArgumentParser(description='Federated MoE')

    # FL settings
    parser.add_argument('--client_nums', default='10,20,30,40,50', type=str,
                        help='the numbers of clients.')

    # Optimizer settings
    parser.add_argument('--algorithms', default='fed_avg,fdlora,fed_moe', type=str,
                        help='optional in ["learned_adaptive_training", "mutual". '
                             '"fed_avg", "fed_prompt", "fed_ptuning", "fed_moe"]')
    parser.add_argument('--do_ft', default=False, type=strtobool,
                        help='whether to finetune the global model after aggregation.')

    # Model settings
    parser.add_argument('--model_name', type=str, required=True)

    # Data settings
    parser.add_argument('--client_dataset_name', type=str, required=True,
                        help='the name of the client dataset.')

    # Other settings
    parser.add_argument('--python_path', default='/home/zyc/miniconda3/bin/python', type=str,
                        help='the path of python interpreter.')
    parser.add_argument('--gpus', default='0,1,2', type=str,
                        help='the indices of GPUs to use.')
    parser.add_argument('--max_retry_times', default=2, type=int)

    args = parser.parse_args()

    algorithms = args.algorithms.split(",")
    client_nums = [int(client_num) for client_num in args.client_nums.split(',')]
    gpus = Queue(maxsize=8)
    for gpu in args.gpus.split(','):
        gpus.put(gpu)

    threads = []
    for client_num, algorithm in itertools.product(client_nums, algorithms):
        time_stamp = int(time.time())
        thread_args = {k: v for k, v in args.__dict__.items() if v is not None}
        del thread_args['python_path']
        del thread_args['max_retry_times']
        del thread_args['algorithms']
        del thread_args['client_nums']
        del thread_args['gpus']
        thread_args['client_num'] = client_num
        thread_args['data_he'] = 'meta1' if args.client_dataset_name == 'natural-instruct' else 'dir1.0'
        thread_args['client_step'] = 200
        thread_args['seed'] = 42
        thread_args['time_stamp'] = time_stamp
        thread_args['log_file'] = True
        thread_args['log_root'] = os.path.join(
            os.path.dirname(__file__), '../logs',
            args.client_dataset_name, algorithm,
            str(client_num), str(time_stamp)
        )

        if algorithm == 'fed_moe':
            del thread_args['do_ft']
            thread_args['expert_choices'] = 4
            train_script = os.path.join(os.path.dirname(__file__), '../main.py')
        else:
            if algorithm == 'fdlora':
                del thread_args['do_ft']
            thread_args['algorithm'] = algorithm
            train_script = os.path.join(os.path.dirname(__file__), '../train_baselines.py')

        t = Thread(
            target=run_train_process,
            args=(
                algorithm, args.python_path, train_script,
                args.max_retry_times, gpus, thread_args
            )
        )
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()

    print("All finished")
