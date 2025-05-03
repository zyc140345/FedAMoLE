import subprocess
import os
import argparse
import time
import itertools
from threading import Thread
from queue import Queue
from typing import Dict


def run_train_process(
    python_path: str,
    main_file: str,
    max_retry_times: int,
    gpus: Queue,
    kwargs: Dict[str, str]
):
    gpu = gpus.get()
    kwargs['gpu'] = gpu

    print("{} expert_num{} expert_choices{} max_experts{} lr{} started".format(
        kwargs['data_he'],
        kwargs['expert_num'],
        kwargs['expert_choices'],
        kwargs['max_experts'],
        kwargs['lr']
    ))
    process_args = [python_path, '-u', main_file]
    for k, v in kwargs.items():
        process_args.extend(['--' + k, str(v)])
    result = subprocess.run(process_args)

    retry_times = 0
    while result.returncode != 0 and retry_times < max_retry_times:
        retry_times += 1
        print("{} expert_num{} expert_choices{} max_experts{} lr{} failed, retrying {}/{}".format(
            kwargs['data_he'],
            kwargs['expert_num'],
            kwargs['expert_choices'],
            kwargs['max_experts'],
            kwargs['lr'],
            retry_times,
            max_retry_times
        ))
        result = subprocess.run(process_args)

    if result.returncode == 0:
        print("{} expert_num{} expert_choices{} max_experts{} lr{} finished".format(
            kwargs['data_he'],
            kwargs['expert_num'],
            kwargs['expert_choices'],
            kwargs['max_experts'],
            kwargs['lr']
        ))
    else:
        print("{} expert_num{} expert_choices{} max_experts{} lr{} failed, max retry times reached".format(
            kwargs['data_he'],
            kwargs['expert_num'],
            kwargs['expert_choices'],
            kwargs['max_experts'],
            kwargs['lr']
        ))

    gpus.put(gpu)


if __name__ == '__main__':
    # ===== Parse arguments =====
    parser = argparse.ArgumentParser(description='Federated MoE')

    # Model settings
    parser.add_argument('--model_name', type=str, required=True)

    # MoE settings
    parser.add_argument('--expert_nums', default='15,20,25,30', type=str,
                        help='Total numbers of domain experts per module.')
    parser.add_argument('--expert_choices', default='2,3,4,5', type=str,
                        help='Numbers of clients selected by each domain expert.')
    parser.add_argument('--max_experts', default='8,10,12,14', type=str,
                        help='Maximum numbers of domain experts assigned to each client module.')

    # Optimizer settings
    parser.add_argument('--lrs', default='5e-4,1e-4,5e-5,1e-5', type=str,
                        help='Learning rates for local fine-tuning.')

    # Data settings
    parser.add_argument('--client_dataset_name', type=str, required=True,
                        help="Name of the client's local dataset.")
    parser.add_argument('--data_hes', type=str, required=True,
                        help='Heterogeneity settings of client data distributions, '
                             'options: ["iid", "dir<alpha>", "meta<class_per_client>"].')

    # Other settings
    parser.add_argument('--python_path', default='/home/zyc/miniconda3/bin/python', type=str,
                        help='The path to the python interpreter.')
    parser.add_argument('--gpus', default='0,1,2', type=str,
                        help='GPU indices to use.')
    parser.add_argument('--max_retry_times', default=2, type=int,
                        help='Maximum number of retries for failed training jobs.')

    args = parser.parse_args()

    expert_nums = [int(expert_num) for expert_num in args.expert_nums.split(',')]
    expert_choices = [int(c) for c in args.expert_choices.split(',')]
    max_experts = [int(m) for m in args.max_experts.split(',')]
    lrs = [float(lr) for lr in args.lrs.split(',')]
    data_hes = args.data_hes.split(',')
    gpus = Queue(maxsize=8)
    for gpu in args.gpus.split(','):
        gpus.put(gpu)

    threads = []
    for expert_num, c, m, lr, data_he in itertools.product(
        expert_nums, expert_choices, max_experts, lrs, data_hes
    ):
        time_stamp = int(time.time())
        thread_args = {k: v for k, v in args.__dict__.items() if v is not None}
        del thread_args['python_path']
        del thread_args['max_retry_times']
        del thread_args['expert_nums']
        del thread_args['lrs']
        del thread_args['data_hes']
        del thread_args['gpus']
        thread_args['expert_num'] = expert_num
        thread_args['expert_choices'] = c
        thread_args['max_experts'] = m
        thread_args['lr'] = lr
        thread_args['data_he'] = data_he
        thread_args['seed'] = 42
        thread_args['time_stamp'] = time_stamp
        thread_args['client_step'] = 200
        thread_args['log_file'] = True
        thread_args['log_root'] = os.path.join(
            os.path.dirname(__file__), '../logs',
            args.client_dataset_name, data_he,
            str(expert_num), str(c), str(m),
            str(lr), str(time_stamp)
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
