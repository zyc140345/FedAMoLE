import os
import argparse
import time
import sys
import json
import shutil
import numpy as np
import torch

from datetime import date
from util import set_seed, get_elapsed_time
from data.data_loader import ClientDataLoader
from data.dataset import load_dataset, DATASET2TASK_TYPE
from node import Client, Server
from torch.utils.tensorboard import SummaryWriter
from distutils.util import strtobool


if __name__ == '__main__':
    # ===== Parse and set arguments =====
    today = date.today()
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description='FedAMoLE')

    # FL settings
    parser.add_argument('--client_num', default=10, type=int,
                        help='Total number of clients.')
    parser.add_argument('--rounds', default=30, type=int,
                        help='Number of communication rounds.')
    parser.add_argument('--local_epochs', default=1, type=int,
                        help='Number of epochs each client performs during local fine-tuning in one round.')
    parser.add_argument('--client_step', default=None, type=int,
                        help='Number of steps each client performs during local fine-tuning in one round.')

    # Model settings
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--precision', default='fp16', type=str,
                        help='Backbone precision, options: ["fp16", "fp32"].')
    parser.add_argument('--lora_rank', default=8, type=int,
                        help='Rank of the LoRA adapter.')
    parser.add_argument('--lora_alpha', default=16, type=float,
                        help='Alpha parameter of the LoRA adapter.')
    parser.add_argument('--lora_dropout', default=0.05, type=float,
                        help='Dropout rate for the LoRA adapter.')

    # FedMoE settings
    parser.add_argument('--expert_num', default=30, type=int,
                        help='Total number of domain experts per module.')
    parser.add_argument('--expert_choices', default=2, type=int,
                        help='Number of clients selected by each domain expert.')
    parser.add_argument('--max_experts', default=8, type=int,
                        help='Maximum number of domain experts assigned to each client module.')
    parser.add_argument('--load_balance_alpha', default=0.001, type=float,
                        help='Weighting coefficient of the load balancing loss.')
    parser.add_argument('--top_k', default=2, type=int,
                        help='Number of domain experts (excluding the shared expert) each token routed to.')
    parser.add_argument('--random_dispatch', default=False, type=strtobool,
                        help='Whether to randomly assign experts to clients. (for ablation study)')
    parser.add_argument('--save_embs', default=False, type=strtobool,
                        help='Whether to save embeddings of the client data and domain experts.')
    parser.add_argument('--save_dispatch', default=False, type=strtobool,
                        help='Whether to save the expert dispatching results.')
    parser.add_argument('--test_init', default=False, type=strtobool,
                        help='Whether to test the initial model.')
    parser.add_argument('--static_arch', default=False, type=strtobool,
                        help='Whether to use static model architectures throughout the FL progress.')
    parser.add_argument('--homo_arch', default=False, type=strtobool,
                        help='Whether to use homogeneous model architectures across clients.')
    parser.add_argument('--apply_dp', default=False, type=strtobool,
                        help='Whether to apply differential privacy.')
    parser.add_argument('--dp_eta', default=100, type=float,
                        help='Privacy budget for differential privacy.')

    # Optimizer settings
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Learning rate for local fine-tuning.')
    parser.add_argument('--lr_decay', default=0.99, type=float,
                        help='Learning rate decay per round.')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for local fine-tuning.')

    # Data settings
    parser.add_argument('--client_dataset_name', type=str, required=True,
                        help="Name of the client's local dataset.")
    parser.add_argument('--client_dataset_ratio', default=1.0, type=float,
                        help='Proportion of the global dataset used for heterogeneous partitioning.')
    parser.add_argument('--data_he', type=str, required=True,
                        help='Heterogeneity setting of client data distributions, '
                             'options: ["iid", "dir<alpha>", "meta<class_per_client>"].')
    parser.add_argument('--ratio_train_to_aux', default='0.95,0.05', type=str,
                        help='Ratio of training set to embedding set.')
    parser.add_argument('--ratio_eval', default=0.1, type=float,
                        help='Proportion of the evaluation set relative to the entire client dataset.')
    parser.add_argument('--max_length', default=1024, type=int,
                        help='Maximum sequence length.')

    # Other settings
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--log_root', default=None, type=str,
                        help='Root directory for logging.')
    parser.add_argument('--log_file', default=False, type=strtobool,
                        help='Whether to redirect stdout and stderr to a log file.')
    parser.add_argument('--log_level', default='summarized', type=str,
                        help='Logging level, options: ["summarized", "detailed"].')
    parser.add_argument('--metric', default='bmta', type=str,
                        help='Model evaluation metric, options: ["last", "bmta"].')
    parser.add_argument('--seed', default=int(str(today.year) + str(today.month) + str(today.day)), type=int,
                        help='Random seed for reproducibility.')
    parser.add_argument('--partition_seed', default=None, type=int,
                        help='Random seed for heterogeneous data partition.')
    parser.add_argument('--time_stamp', default=int(time.time()), type=int,
                        help='Current timestamp. (to distinguish between different runs)')
    parser.add_argument('--do_profile', default=False, type=strtobool,
                        help='Whether to monitor memory usage and time consumption.')

    args = parser.parse_args()

    # set seed for data partition
    if args.partition_seed is not None:
        set_seed(args.partition_seed)
    else:
        set_seed(args.seed)

    # set default log root
    algorithm = 'fed_moe'
    if args.random_dispatch:
        algorithm += '_random'
    if args.static_arch:
        algorithm += '_static'
    if args.homo_arch:
        algorithm += '_homo'
    if args.log_root is None:
        args.log_root = os.path.join(
            os.path.dirname(__file__), 'logs', args.client_dataset_name, args.data_he,
            algorithm, str(args.seed), str(args.time_stamp)
        )

    # make log directory
    if os.path.exists(args.log_root):
        shutil.rmtree(args.log_root)
    os.makedirs(args.log_root)
    if args.save_embs:
        os.makedirs(os.path.join(args.log_root, 'embs'))
    if args.save_dispatch:
        os.makedirs(os.path.join(args.log_root, 'dispatch'))

    # validate the MoE settings
    if args.expert_num * args.expert_choices > args.client_num * args.max_experts:
        raise ValueError(
            f'The total clients chosen by experts ({args.expert_num} * {args.expert_choices})'
            f' exceeds the total expert volume ({args.client_num} * {args.max_experts})!'
        )

    # save the arguments
    with open(os.path.join(args.log_root, 'params.json'), 'w') as f:
        json.dump({k: v for k, v in args.__dict__.items()}, f)

    # redirect stdout and stderr to log file
    log_file = None
    if args.log_file:
        log_file = open(os.path.join(args.log_root, 'train.log'), 'w')
        sys.stdout = log_file
        sys.stderr = log_file

    # set the task type
    args.task_type = DATASET2TASK_TYPE[args.client_dataset_name]

    # ===== Construct data loader =====
    print(f'=============== Loading Dataset ===============')
    dataset, ratios, label_encoder = load_dataset(
        args.client_dataset_name,
        down_sample_rate=args.client_dataset_ratio,
        ratio_train_to_aux=args.ratio_train_to_aux,
        ratio_eval=args.ratio_eval
    )
    data_loader = ClientDataLoader(dataset, ratios, label_encoder, args)

    # set seed for federated learning
    if args.partition_seed is not None:
        set_seed(args.seed)

    # ===== Construct clients and server =====
    suffix = f'.{args.client_dataset_name}_{args.data_he}_{args.seed}_{args.time_stamp}'
    writer = SummaryWriter(
        log_dir=os.path.join(args.log_root, "summary"),
        filename_suffix=suffix, flush_secs=1
    )
    server = Server(data_loader.tokenizer, writer, args)
    clients = []
    for client_idx in range(args.client_num):
        client = Client(
            idx=client_idx,
            train_loader=data_loader.train_loaders[client_idx],
            aux_loader=data_loader.aux_loaders[client_idx],
            eval_loader=data_loader.eval_loaders[client_idx],
            test_loader=data_loader.test_loaders[client_idx],
            tokenizer=data_loader.tokenizer,
            writer=writer, args=args
        )
        clients.append(client)
    server.register_clients(clients)

    # ===== Initialize metrics =====
    eval_loss = [0] * (args.rounds + 1)
    test_loss = [0] * (args.rounds + 1)
    test_acc = [0] * (args.rounds + 1)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    train_time = [[0] * args.client_num for _ in range(args.rounds)]
    test_time = [[0] * args.client_num for _ in range(args.rounds + 1)]
    aggregate_time = [0] * (args.rounds - 1)
    if args.do_profile:
        torch.cuda.reset_peak_memory_stats(args.gpu)

    # ===== Federated learning =====
    server.dispatch_experts(0)
    for r in range(0, args.rounds + 1):
        if r > 0 or (r == 0 and args.test_init):  # Round 0 is for testing the initial model
            print(f'\n=============== The {r}-th round ===============')
            for client_idx, client in enumerate(clients):
                client.load_model()

                if r > 0:  # local fine-tuning
                    if args.do_profile:
                        start.record()
                    client.train_experts(r)
                    if args.do_profile:
                        train_time[r - 1][client_idx] = get_elapsed_time(start, end)

                if not args.save_dispatch:
                    eval_loss[r] += client.eval()  # eval

                    if args.metric == "bmta" or r == args.rounds:  # test
                        if args.do_profile:
                            start.record()
                        metrics = client.test(r)
                        if args.do_profile:
                            test_time[r][client_idx] = get_elapsed_time(start, end)
                        test_loss[r] += metrics[0]
                        test_acc[r] += metrics[1]

                client.release_model()

        eval_loss[r] /= args.client_num
        writer.add_scalar("Loss/Eval/Round", eval_loss[r], r)

        if args.metric == "bmta" or r == args.rounds:
            test_loss[r] /= args.client_num
            writer.add_scalar("Loss/Test/Round", test_loss[r], r)
            if args.task_type == 'cls_lm':
                test_acc[r] /= args.client_num
                writer.add_scalar("Acc/Test/Round", test_acc[r], r)

        if 0 < r < args.rounds:
            for client in clients:  # estimate embeddings
                client.load_model()
                if args.do_profile:
                    start.record()
                if not args.static_arch:
                    client.compute_embs(r)
                if args.do_profile:
                    aggregate_time[r - 1] = max(aggregate_time[r - 1], get_elapsed_time(start, end))
                client.release_model()

            # aggregate and dispatch experts
            if args.do_profile:
                start.record()
            server.aggregate()
            server.dispatch_experts(r)
            if args.do_profile:
                aggregate_time[r - 1] += get_elapsed_time(start, end)

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

    if args.task_type == "causal_lm":
        best_r = np.argmax(test_loss)
        print(f'Best Round {best_r}: avg_loss={test_loss[best_r]:.4f}')
    elif args.task_type == "cls_lm":
        best_r = np.argmax(test_acc)
        print(f'Best Round {best_r}: avg_loss={test_loss[best_r]:.4f}, avg_acc={test_acc[best_r] * 100:.2f}%')

    if args.log_file:
        log_file.close()
