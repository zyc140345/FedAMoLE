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
    python_path: str,
    main_file: str,
    max_retry_times: int,
    gpus: Queue,
    kwargs: Dict[str, str]
):
    gpu = gpus.get()
    kwargs['gpu'] = gpu

    print(f"fed_moe eta={kwargs['dp_eta']} started")
    process_args = [python_path, '-u', main_file]
    for k, v in kwargs.items():
        process_args.extend(['--' + k, str(v)])
    result = subprocess.run(process_args)

    retry_times = 0
    while result.returncode != 0 and retry_times < max_retry_times:
        retry_times += 1
        print(f"fed_moe eta={kwargs['dp_eta']} failed, retrying {retry_times}/{max_retry_times}")
        result = subprocess.run(process_args)

    if result.returncode == 0:
        print(f"fed_moe eta={kwargs['dp_eta']} finished")
    else:
        print(f"fed_moe eta={kwargs['dp_eta']} failed, max retry times reached")

    gpus.put(gpu)


if __name__ == '__main__':
    # ===== Parse arguments =====
    parser = argparse.ArgumentParser(description='Federated MoE')

    # Model settings
    parser.add_argument('--model_name', type=str, required=True)

    # MoE settings
    parser.add_argument('--dp_etas', default='1,50,100', type=str,
                        help='Privacy budgets for differential privacy.')

    # Data settings
    parser.add_argument('--client_dataset_name', type=str, required=True,
                        help="Name of the client's local dataset.")

    # Other settings
    parser.add_argument('--python_path', default='/home/zyc/miniconda3/bin/python', type=str,
                        help='The path to the python interpreter.')
    parser.add_argument('--gpus', default='0,1,2', type=str,
                        help='GPU indices to use.')
    parser.add_argument('--seeds', default='42,62,82', type=str,
                        help='Random seeds for reproducibility.')
    parser.add_argument('--max_retry_times', default=2, type=int,
                        help='Maximum number of retries for failed training jobs.')

    args = parser.parse_args()

    dp_etas = args.dp_etas.split(',')
    gpus = Queue(maxsize=8)
    seeds = args.seeds.split(',')
    for gpu in args.gpus.split(','):
        gpus.put(gpu)

    threads = []
    for dp_eta, seed in itertools.product(dp_etas, seeds):
        time_stamp = int(time.time())
        thread_args = {k: v for k, v in args.__dict__.items() if v is not None}
        del thread_args['python_path']
        del thread_args['max_retry_times']
        del thread_args['dp_etas']
        del thread_args['gpus']
        del thread_args['seeds']
        thread_args['data_he'] = 'meta1' if args.client_dataset_name == 'natural-instruct' else 'dir1.0'
        thread_args['client_step'] = 200
        thread_args['apply_dp'] = True
        thread_args['dp_eta'] = dp_eta
        thread_args['seed'] = seed
        thread_args['time_stamp'] = time_stamp
        thread_args['log_file'] = True
        thread_args['log_root'] = os.path.join(
            os.path.dirname(__file__), '../logs',
            args.client_dataset_name, str(dp_eta),
            str(seed), str(time_stamp)
        )

        train_script = os.path.join(os.path.dirname(__file__), '../main.py')
        t = Thread(
            target=run_train_process,
            args=(
                args.python_path, train_script,
                args.max_retry_times, gpus, thread_args
            )
        )
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()

    print("All finished")
