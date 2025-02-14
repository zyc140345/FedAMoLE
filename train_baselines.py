import os
import argparse
import torch
import time
import shutil
import sys
import numpy as np

from data.data_loader import ClientDataLoader
from data.dataset import load_dataset, DATASET2TASK_TYPE
from baselines.recorder import Recorder
from baselines.evaluator import Evaluator
from baselines.node import Server, Client
from baselines.trainer import Trainer
from datetime import date
from util import set_seed, get_elapsed_time
from distutils.util import strtobool


if __name__ == '__main__':
    today = date.today()
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description='Baselines')

    # FL settings
    parser.add_argument('--client_num', default=10, type=int,
                        help='Total number of clients')
    parser.add_argument('--rounds', '-r', default=30, type=int,
                        help='Number of communication rounds.')
    parser.add_argument('--local_epochs', default=1, type=int,
                        help='Number of epochs each client performs during local fine-tuning in one round.')
    parser.add_argument('--client_step', default=None, type=int,
                        help='Number of steps each client performs during local fine-tuning in one round.')
    parser.add_argument('--do_ft', default=False, type=strtobool,
                        help='Whether to fine-tune the global model after aggregation.')

    # Model settings
    parser.add_argument('--model_name', default='opt-350m', type=str)
    parser.add_argument('--precision', default='fp16', type=str,
                        help='Backbone precision, options: ["fp16", "fp32"].')
    parser.add_argument('--lora_rank', default=8, type=int,
                        help='Rank of the LoRA adapter.')
    parser.add_argument('--lora_alpha', default=16, type=int,
                        help='Alpha parameter of the LoRA adapter.')
    parser.add_argument('--lora_dropout', default=0.05, type=float,
                        help='Dropout rate for the LoRA adapter.')

    # Optimizer settings
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Learning rate for local fine-tuning.')
    parser.add_argument('--lr_decay', default=0.99, type=float,
                        help='Learning rate decay per round.')
    parser.add_argument('--batch_size', '-b', default=1, type=int,
                        help='Batch size for local fine-tuning.')
    parser.add_argument('--algorithm', default='fed_avg', type=str,
                        help='Algorithm to use, options: ["learned_adaptive_training", "mutual", '
                             '"fed_avg", "fed_prompt", "fed_ptuning", "fdlora"]')

    # Data settings
    parser.add_argument('--client_dataset_name', '-d', default='dolly-15k', type=str,
                        help="Name of the client's local dataset.")
    parser.add_argument('--client_dataset_ratio', default=1.0, type=float,
                        help='Proportion of the global dataset used for heterogeneous partitioning.')
    parser.add_argument('--data_he', type=str, required=True,
                        help='Heterogeneity setting of client data distributions, '
                             'options: ["iid", "dir<alpha>", "meta<class_per_client>"].')
    parser.add_argument('--ratio_train_to_aux', default='0.95,0.05', type=str,
                        help='Ratio of training set to adaptability set. (for FedAPEN)')
    parser.add_argument('--ratio_eval', default=0.1, type=float,
                        help='Proportion of the evaluation set relative to the entire client dataset.')
    parser.add_argument('--max_length', default=1024, type=int,
                        help='Maximum sequence length.')

    # FDLORA settings
    parser.add_argument('--sync_freq', default=5, type=int,
                        help='Frequency of synchronizing the shared and private adapters.')

    # Other settings
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--log_root', default='none', type=str,
                        help='Root directory for logging.')
    parser.add_argument('--log_file', default=False, type=bool,
                        help='Whether to redirect stdout and stderr to a log file.')
    parser.add_argument('--log_level', default='summarized', type=str,
                        help='Logging level, options: ["summarized", "detailed"].')
    parser.add_argument('--metric', default='bmta', type=str,
                        help='Model evaluation metric, options: ["last", "bmta"].')
    parser.add_argument('--seed', default=int(str(today.year) + str(today.month) + str(today.day)), type=int,
                        help='Random seed for reproducibility.')
    parser.add_argument('--time_stamp', default=int(time.time()), type=int,
                        help='Current timestamp. (to distinguish between different runs)')
    parser.add_argument('--do_profile', default=False, type=strtobool,
                        help='Whether to monitor memory usage and time consumption.')

    args = parser.parse_args()
    torch.cuda.set_device(torch.device(f'cuda:{args.gpu}'))

    # FedAPEN require adaptability set
    if args.algorithm not in ["learned_adaptive_training", "fdlora"]:
        args.ratio_train_to_aux = '1.0,0.0'

    # set task type
    args.task_type = DATASET2TASK_TYPE[args.client_dataset_name]

    # set seed for reproducibility
    set_seed(args.seed)

    # set default log root
    if args.log_root == 'none':
        args.log_root = os.path.join(
            os.path.dirname(__file__), 'logs', args.client_dataset_name,
            args.data_he, args.algorithm, str(args.seed), str(args.time_stamp)
        )

    # make log directory
    if os.path.exists(args.log_root):
        shutil.rmtree(args.log_root)
    os.makedirs(args.log_root)

    # save the arguments
    with open(os.path.join(args.log_root, 'params.log'), 'w') as writer:
        for k, v in args.__dict__.items():
            print(k, ':', v, file=writer)

    # redirect stdout and stderr to log file
    log_file = None
    if args.log_file:
        log_file = open(os.path.join(args.log_root, 'train.log'), 'w')
        sys.stdout = log_file
        sys.stderr = log_file

    # ===== Construct data loader =====
    dataset, ratios, label_encoder = load_dataset(
        args.client_dataset_name,
        down_sample_rate=args.client_dataset_ratio,
        ratio_train_to_aux=args.ratio_train_to_aux,
        ratio_eval=args.ratio_eval
    )
    dl = ClientDataLoader(dataset, ratios, label_encoder, args)

    # ===== Construct clients and server =====
    evaluator = Evaluator(args)
    recorder = Recorder(args, evaluator)
    trainer = Trainer(recorder, args)
    server = Server(recorder=recorder, tokenizer=dl.tokenizer, args=args)
    clients = [
        Client(
            idx=idx, args=args,
            train_loader=dl.train_loaders[idx],
            aux_loader=dl.aux_loaders[idx],
            eval_loader=dl.eval_loaders[idx],
            test_loader=dl.test_loaders[idx],
            tokenizer=dl.tokenizer
        )
        for idx in range(args.client_num)
    ]
    recorder.register_clients(clients)
    server.aggregate(clients, cur_round=-2)  # first initialize all the shared model at the same state

    # ===== Initialize metrics =====
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats(args.gpu)
    train_time = [[0] * args.client_num for _ in range(args.rounds)]
    test_time = [[0] * args.client_num for _ in range(args.rounds)]
    aggregate_time = [0] * args.rounds

    # ===== FDLoRA Local Learning =====
    if args.algorithm.lower() == 'fdlora':
        print('\n===============Local Training===============')
        for client in clients:
            client.load_models()
            trainer(client, -1)
            client.release_models()
        server.aggregate(clients, cur_round=-1)

    # ===== Federated learning =====
    print('\nAlgorithm: {0}'.format(args.algorithm))
    for cur_round in range(args.rounds):
        print('\n===============The {:d}-th round==============='.format(cur_round + 1))
        for i, client in enumerate(clients):
            client.load_models()
            if args.do_profile:
                start.record()
            trainer(client, cur_round)  # local fine-tuning
            if args.do_profile:
                train_time[cur_round][i] = get_elapsed_time(start, end)
            # for methods based on post-aggregation fine-tuning, we should test the model after local fine-tuning
            if args.do_ft and (args.metric == 'bmta' or cur_round == args.rounds - 1):
                recorder.eval(client)
                recorder.test(client)
            client.release_models()

        if args.do_profile:
            start.record()
        server.aggregate(clients, cur_round)
        if args.do_profile:
            aggregate_time[cur_round] = get_elapsed_time(start, end)

        for i, client in enumerate(clients):
            client.load_models(enable=args.algorithm.lower() != 'fdlora')
            if args.algorithm.lower() == 'learned_adaptive_training':  # for FedAPEN, we need to optimize lambda
                if args.do_profile:
                    start.record()
                client.train_lam()
                if args.do_profile:
                    train_time[cur_round][i] += get_elapsed_time(start, end)
            if args.algorithm.lower() == 'fdlora':  # for FDLoRA, we need to perform adaptive fusion
                if args.do_profile:
                    start.record()
                client.adaptive_fusion()
                if args.do_profile:
                    test_time[cur_round][i] = get_elapsed_time(start, end)
            if args.metric == 'bmta' or cur_round == args.rounds - 1:
                recorder.eval(client)
                if args.do_profile:
                    start.record()
                recorder.test(client)
                if args.do_profile:
                    test_time[cur_round][i] += get_elapsed_time(start, end)
            client.release_models(enable=args.algorithm.lower() != 'fdlora')

        if args.metric == 'bmta' or cur_round == args.rounds - 1:
            recorder.summary(cur_round)

    if args.do_profile:
        train_time = np.mean([np.max(t) / 1000 for t in train_time])
        test_time = np.mean([np.max(t) / 1000 for t in test_time])
        aggregate_time = np.mean(aggregate_time) / 1000
        peak_memory = torch.cuda.max_memory_allocated(args.gpu) / 1024 / 1024
        upload_volume = np.mean([
            np.mean(client.upload_volume) / 1024 / 1024 for client in clients
        ])
        download_volume = np.mean([
            np.mean(client.download_volume) / 1024 / 1024 for client in clients
        ])
        print(f'\n=============== Profiling ===============')
        print(f'Average Train Time: {train_time:.2f} s')
        print(f'Average Test Time: {test_time:.2f} s')
        print(f'Average Aggregate Time: {aggregate_time:.2f} s')
        print(f'Peak Memory: {peak_memory:.2f} MB')
        print(f'Average Upload Volume: {upload_volume:.2f} MB')
        print(f'Average Download Volume: {download_volume:.2f} MB')

    if args.log_file:
        log_file.close()
