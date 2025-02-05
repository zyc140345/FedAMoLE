import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import platform
import evaluation
import json
import torch
import pickle
import math
import re
import pylab

from sklearn.manifold import TSNE
from data.dataset import load_dataset, DATASET2TASK_TYPE
from data.hete_partition import partition_idx_dir
from util import set_seed, list_dir
from evaluation import parse_tensorboard_log, parse_json_log, mta, mtr
from matplotlib.collections import LineCollection
from util import set_seed

ALG_MAP = {
    'fed_avg': 'FedIT',
    'fed_avg_tune': 'FedIT-FT',
    'fed_prompt': 'FedPrompt',
    'fed_prompt_tune': 'FedPrompt-FT',
    'fed_ptuning': 'FedPTuning',
    'fed_ptuning_tune': 'FedPTuning-FT',
    'fdlora': 'FDLoRA',
    'fed_moe': 'FedAMoLE',
    'fed_moe_vanilla_router_random': 'FedAMoLE (vanilla router, random)',
    'fed_moe_vanilla_router': 'FedAMoLE (vanilla router)',
    'fed_moe_random': 'FedAMoLE (random)'
}

DATASET_MAP = {
    'rte': 'RTE',
    'wic': 'WiC',
    'boolq': 'BoolQ',
    'dolly-15k': 'Dolly-15k',
    'natural-instruct': 'Natural Instruct',
    'alpaca': 'Alpaca',
    'squad': 'SQuAD',
    'ag-news': 'AG News',
    'race': 'RACE',
    'snli': 'SNLI',
    'mnli': 'MNLI',
    'yelp': 'YELP',
    'yelp-p': 'YELP-P'
}

COLORS = ['#D96248', '#554687', '#F2A81D', '#285947', '#9B41D1', '#72DB1A', '#342B56', '#4A96D9', '#F22E8A', '#FF0200']
MARKERS = ['8', 's', 'p', '<', 'D', '>', 'P', 'h', '*', 'X']


def plot_legend(legend_elements, save_dir=None, file_name=None):
    legend_fig = pylab.figure(figsize=(8, 1))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis('off')
    legend = legend_ax.legend(handles=legend_elements, loc='center', ncol=len(legend_elements),
                              frameon=True, fontsize=28)

    legend_fig.canvas.draw()
    bbox = legend.get_window_extent(legend_fig.canvas.get_renderer())
    bbox = bbox.transformed(legend_fig.dpi_scale_trans.inverted())
    legend_fig.set_size_inches(bbox.width, bbox.height)

    if save_dir is not None and file_name is not None:
        legend_path = os.path.join(save_dir, file_name)
        legend_fig.savefig(legend_path, bbox_inches='tight', pad_inches=0.1, backend='pgf')


def plot_data_distribution(
    dataset_name, down_sample_rate,
    client_num, alphas, seed, n_col=3, x_step=10,
    subplot_size=(8, 4), save_dir=None
):
    set_seed(seed)
    dataset = load_dataset(dataset_name, down_sample_rate=down_sample_rate)
    dataset = dataset[0]

    n_row = (len(alphas) - 1) // n_col + 1
    fig = plt.figure(figsize=(n_col * subplot_size[0], n_row * subplot_size[1]))
    gs = gridspec.GridSpec(n_row, n_col, figure=fig)

    y = np.array([item["class"] if isinstance(item, dict) else item[1] for item in dataset])
    for i, alpha in enumerate(alphas):
        r = i // n_col
        c = i % n_col
        ax = fig.add_subplot(gs[r, c])

        num_classes = np.unique(y).shape[0]
        idx_parts = partition_idx_dir(y, client_num, alpha=alpha, num_classes=num_classes)
        class_counts = np.array([np.bincount(y[idx_part], minlength=num_classes) for idx_part in idx_parts])

        sns.heatmap(class_counts, ax=ax, annot=False, fmt='d', cmap='Blues', cbar=True)

        ax.set_title(f'$({chr(i + 97)})\\ \\alpha={alpha}$')
        ax.set_xlabel('Class ID')
        if num_classes < 10:
            ax.set_xticks(np.arange(num_classes) + 0.5)
            ax.set_xticklabels(np.arange(num_classes) + 1)
        else:
            tick_positions = np.array([0] + [i * x_step + x_step - 1 for i in range(num_classes // x_step)]) + 0.5
            ax.set_xticks(tick_positions)
            ax.set_xticklabels((tick_positions + 0.5).astype(int))

        ax.set_ylabel('Client ID')
        ax.set_yticks(np.arange(client_num) + 0.5)
        ax.set_yticklabels(np.arange(1, client_num + 1))

    plt.tight_layout()
    plt.show()

    if save_dir is not None:
        save_path = os.path.join(save_dir, f'{dataset_name}.pdf')
        fig.savefig(save_path, backend='pgf')


def plot_metric_per_round(root_dir, save_dir=None, n_col=2, last=False,
                          error_bar=False, round_step=1, sep_legend=False):
    result = evaluation.summary(root_dir, last=last, return_all=True, include_ft=True)

    for dataset_id, dataset in enumerate(result.keys()):
        task_type = DATASET2TASK_TYPE[dataset]
        metric = 'rouge' if task_type == 'causal_lm' else 'acc'

        n_iids = len(result[dataset])
        n_row = (n_iids - 1) // n_col + 1
        fig = plt.figure(figsize=(n_col * 8, n_row * 6))
        gs = gridspec.GridSpec(n_row, n_col, figure=fig)

        legend_elements = []

        iids = list(result[dataset].keys())
        sort_idx = np.argsort([
            float(iid[3:]) if iid.startswith('dir') else float(iid[4:])
            for iid in iids
        ])
        iids = [iids[i] for i in sort_idx]
        for i, iid in enumerate(iids):
            r = i // n_col
            c = i % n_col
            ax = fig.add_subplot(gs[r, c])

            init_data = result[dataset][iid]['fed_moe'][:, :1]
            for j, (algorithm, data) in enumerate(result[dataset][iid].items()):
                marker = MARKERS[j % len(MARKERS)]
                color = COLORS[j % len(COLORS)]
                if algorithm != 'fed_moe':
                    data = np.hstack([init_data, data])

                if len(data.shape) == 1:
                    rounds = range(0, len(data), round_step)
                    data = data[::round_step]
                    ax.plot(rounds, data, label=ALG_MAP[algorithm],
                            marker=marker, markersize=12, markerfacecolor='none',
                            markeredgewidth=2, linewidth=2, color=color)
                elif error_bar:
                    mean_metric = np.mean(data, axis=0)
                    rounds = range(0, len(mean_metric), round_step)
                    mean_metric = mean_metric[::round_step]
                    max_metric = np.max(data, axis=0)[::round_step]
                    min_metric = np.min(data, axis=0)[::round_step]
                    line = ax.plot(rounds, mean_metric, label=ALG_MAP[algorithm],
                                   marker=marker, markersize=12, markerfacecolor='none',
                                   markeredgewidth=2, linewidth=2, color=color)
                    line_color = line[0].get_color()
                    ax.fill_between(rounds, min_metric, max_metric, alpha=0.2,
                                    edgecolor=line_color, linewidth=1, facecolor=line_color)
                else:
                    mean_metric = np.mean(data, axis=0)
                    rounds = range(0, len(mean_metric), round_step)
                    mean_metric = mean_metric[::round_step]
                    ax.plot(rounds, mean_metric, label=ALG_MAP[algorithm],
                            marker=marker, markersize=12, markerfacecolor='none',
                            markeredgewidth=2, linewidth=2, color=color)

                if dataset_id == 0:  # only add legend for the first subplot
                    legend_elements.append(plt.Line2D([0], [0], label=ALG_MAP[algorithm], color=color,
                                                      linewidth=2, markeredgewidth=2, marker=marker,
                                                      markersize=24, markerfacecolor='none'))

            ax.set_xlabel(r'Round', fontsize=24)
            ax.set_ylabel('ROUGE-L (%)' if metric == 'rouge' else 'Accuracy (%)', fontsize=24)
            ax.tick_params(labelsize=24)
            if not sep_legend:
                ax.legend(frameon=True, facecolor='white', fontsize=18, ncol=2)
            if n_col > 1:
                if iid.startswith('dir'):
                    ax.set_title(f'$({chr(i + 97)})\\ \\alpha={iid[3:]}$', fontsize=14)
                else:
                    ax.set_title(f'$({chr(i + 97)})$ {iid[4:]} Task per Client', fontsize=14)

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.25, hspace=0.3)
        fig.show()

        if save_dir is not None:
            save_path = os.path.join(save_dir, f'{dataset}.pdf')
            fig.savefig(save_path, backend='pgf')

        if sep_legend:
            plot_legend(legend_elements, save_dir, f'metric_per_round_legend.pdf')


def plot_metric_per_iid(root_dir, save_dir=None, last=False, sep_legend=False):
    result = evaluation.summary(root_dir, last=False, return_all=True)

    def calc_metric(x):
        return x[:, -1] if last else x.max(axis=1)

    legend_elements = []
    for dataset_id, dataset in enumerate(result.keys()):
        fig, ax = plt.subplots()

        task_type = DATASET2TASK_TYPE[dataset]
        metric = 'rouge' if task_type == 'causal_lm' else 'acc'

        iids = list(result[dataset].keys())
        sort_idx = np.argsort([float(iid[3:]) for iid in iids])
        iids = [iids[i] for i in sort_idx]

        algorithms = list(result[dataset][iids[0]].keys())
        for i, algorithm in enumerate(algorithms):
            metrics_per_iid = [calc_metric(result[dataset][iid][algorithm]) for iid in iids]
            mean_metric = np.array([np.mean(acc) for acc in metrics_per_iid])

            color = COLORS[i % len(COLORS)]
            marker = MARKERS[i % len(MARKERS)]

            if len(metrics_per_iid[0]) == 1:
                iid_labels = [f'$\\alpha={iid[3:]}$' for iid in iids]
                plt.plot(iid_labels, mean_metric, label=ALG_MAP[algorithm], marker=marker, markeredgewidth=2,
                         markersize=16, markerfacecolor='none', linewidth=2, color=color)
            else:
                max_metric = np.array([np.max(metric) for metric in metrics_per_iid])
                min_metric = np.array([np.min(metric) for metric in metrics_per_iid])

                iid_labels = [f'{float(iid[3:]):.1f}' for iid in iids]
                plt.plot(iid_labels, mean_metric, label=ALG_MAP[algorithm],
                         marker=marker, markersize=16, markerfacecolor='none',
                         markeredgewidth=2, linewidth=2, color=color)

                xs = np.array(ax.get_xticks()).astype('float')
                x_ls = xs - np.full_like(xs, 0.07)
                x_rs = xs + np.full_like(xs, 0.07)
                up_caps = [[[x_l, y_u], [x_r, y_u]] for x_l, x_r, y_u in zip(x_ls, x_rs, max_metric)]
                down_caps = [[[x_l, y_d], [x_r, y_d]] for x_l, x_r, y_d in zip(x_ls, x_rs, min_metric)]
                line_cols = [[[x, y_d], [x, y_u]] for x, y_d, y_u in zip(xs, min_metric, max_metric)]
                lines = LineCollection(
                    up_caps + down_caps + line_cols,
                    linewidths=2, color=color
                )
                ax.add_collection(lines)

            if dataset_id == 0:  # only add legend for the first subplot
                legend_elements.append(plt.Line2D([0], [0], label=ALG_MAP[algorithm], color=color,
                                                  linewidth=2, markeredgewidth=2, marker=marker,
                                                  markersize=24, markerfacecolor='none'))

        ax.autoscale()
        ax.set_xlabel(r'$\alpha$', fontsize=24)
        ax.set_ylabel('Accuracy (%)' if metric == 'acc' else 'ROUGE-L (%)', fontsize=24)
        ax.tick_params(labelsize=24)
        if not sep_legend:
            ax.legend(frameon=True, facecolor='white', fontsize=20)

        fig.tight_layout()
        fig.show()

        if save_dir is not None:
            save_path = os.path.join(save_dir, f'{dataset}_var_iids.pdf')
            fig.savefig(save_path, backend='pgf')

    if sep_legend:
        plot_legend(legend_elements, save_dir, f'metric_per_iid_legend.pdf')


def plot_metric_per_client_num(root_dir, save_dir=None, last=False, include_ft=False, sep_legend=False):
    def calc_metric(x):
        return x[:, -1] if last else x.max(axis=1)

    legend_elements = []
    datasets = list_dir(root_dir)
    for dataset_id, dataset in enumerate(datasets):
        fig, ax = plt.subplots()

        task_type = DATASET2TASK_TYPE[dataset]
        metric_name = 'rouge' if task_type == 'causal_lm' else 'acc'

        cur_dir = os.path.join(root_dir, dataset)
        algorithms = list_dir(cur_dir)
        i = 0
        for algorithm in algorithms:
            metrics = []
            metrics_tune = []
            client_nums = sorted(list_dir(os.path.join(cur_dir, algorithm)), key=lambda x: int(x))
            for client_num in client_nums:
                log_dir = os.path.join(cur_dir, algorithm, client_num)
                timestamp = list_dir(log_dir)[0]
                log_dir = os.path.join(log_dir, timestamp)
                if algorithm == 'fed_moe':
                    result, params = parse_tensorboard_log(log_dir)
                else:
                    result, params = parse_json_log(log_dir)

                if task_type == 'cls_lm':
                    metric = mta(result, algorithm)
                else:
                    metric = mtr(result, algorithm)

                if metric.shape[0] == 2 * params['rounds']:
                    if include_ft:
                        metric_tune = metric[range(0, metric.shape[0], 2)]
                        metric_tune = metric_tune[np.newaxis, :]
                        metrics_tune.append(calc_metric(metric_tune))
                    metric = metric[range(1, metric.shape[0], 2)]
                    metric = metric[np.newaxis, :]
                    metrics.append(calc_metric(metric))
                else:
                    if len(metric.shape) == 1:
                        metric = metric[np.newaxis, :]
                    metrics.append(calc_metric(metric))

            marker = MARKERS[i % len(MARKERS)]
            color = COLORS[i % len(COLORS)]
            plt.plot(client_nums, metrics, label=ALG_MAP[algorithm],
                     marker=marker, markersize=16, markerfacecolor='none',
                     markeredgewidth=2, linewidth=2, color=color)

            if dataset_id == 0:  # only add legend for the first subplot
                legend_elements.append(plt.Line2D([0], [0], label=ALG_MAP[algorithm], color=color,
                                                  linewidth=2, markeredgewidth=2, marker=marker,
                                                  markersize=24, markerfacecolor='none'))

            if metrics_tune:
                i += 1
                marker = MARKERS[i % len(MARKERS)]
                color = COLORS[i % len(COLORS)]
                plt.plot(client_nums, metrics_tune, label=ALG_MAP[algorithm + '_tune'],
                         marker=marker, markersize=16, markerfacecolor='none',
                         markeredgewidth=2, linewidth=2, color=color)

                if dataset_id == 0:  # only add legend for the first subplot
                    legend_elements.append(plt.Line2D([0], [0], label=ALG_MAP[algorithm + '_tune'], color=color,
                                                      linewidth=2, markeredgewidth=2, marker=marker,
                                                      markersize=24, markerfacecolor='none'))

            i += 1

        ax.autoscale()
        ax.set_xlabel('Client Number', fontsize=24)
        ax.set_ylabel('Accuracy (%)' if metric_name == 'acc' else 'ROUGE-L (%)', fontsize=24)
        ax.tick_params(labelsize=24)
        if not sep_legend:
            ax.legend(frameon=True, facecolor='white', fontsize=18)

        fig.tight_layout()
        fig.show()

        if save_dir is not None:
            save_path = os.path.join(save_dir, f'{dataset}_var_client_num.pdf')
            fig.savefig(save_path, backend='pgf')

    if sep_legend:
        plot_legend(legend_elements, save_dir, f'metric_var_client_num_legend.pdf')


def plot_expert_choices_vs_lr(root_dir: str, save_dir=None, expert_num=15, max_experts=8, last=False):
    dataset = list_dir(root_dir)[0]
    iid = list_dir(os.path.join(root_dir, dataset))[0]
    root_dir = os.path.join(root_dir, dataset, iid, str(expert_num))

    expert_choices = list_dir(root_dir)
    expert_choices = sorted(expert_choices, key=lambda x: int(x))
    task_type = None
    for i, c in enumerate(expert_choices):
        cur_dir = os.path.join(root_dir, c, str(max_experts))
        lrs = sorted(list_dir(cur_dir), key=lambda x: float(x), reverse=True)

        metrics = []
        for lr in lrs:
            timestamp = list_dir(os.path.join(cur_dir, lr))[0]
            log_dir = os.path.join(cur_dir, lr, timestamp)
            result, params = parse_tensorboard_log(log_dir)
            task_type = DATASET2TASK_TYPE[params['client_dataset_name']]
            if task_type == 'cls_lm':
                data = mta(result, 'fed_moe')
            else:
                data = mtr(result, 'fed_moe')
            metric = data[-1] if last else np.max(data)
            metrics.append(metric)

        lrs = [f"{float(lr):.0e}" for lr in lrs]

        marker = MARKERS[i % len(MARKERS)]
        color = COLORS[i % len(COLORS)]

        plt.plot(lrs, metrics, label=f"$k^c={c}$",
                 marker=marker, markersize=16, markerfacecolor='none',
                 markeredgewidth=2, linewidth=2, color=color)

    plt.xlabel(r'$\eta$', fontsize=24)
    plt.ylabel('Accuracy (%)' if task_type == 'cls_lm' else 'ROUGE-L (%)', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=24)
    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, "expert_choices_vs_lr.pdf")
        plt.savefig(save_path, backend='pgf')

    plt.show()


def plot_expert_num_vs_max_experts(root_dir: str, save_dir=None, expert_choices=2, lr=5e-5, last=False):
    dataset = list_dir(root_dir)[0]
    iid = list_dir(os.path.join(root_dir, dataset))[0]
    root_dir = os.path.join(root_dir, dataset, iid)

    expert_nums = list_dir(root_dir)
    expert_nums = sorted(expert_nums, key=lambda x: int(x))
    max_experts = None
    task_type = None
    metrics = [[] for _ in expert_nums]
    for i, expert_num in enumerate(expert_nums):
        cur_dir = os.path.join(root_dir, expert_num, str(expert_choices))
        max_experts = sorted(list_dir(cur_dir), key=lambda x: int(x))

        for m in max_experts:
            timestamp = list_dir(os.path.join(cur_dir, m, str(lr)))[0]
            log_dir = os.path.join(cur_dir, m, str(lr), timestamp)
            result, params = parse_tensorboard_log(log_dir)
            task_type = DATASET2TASK_TYPE[params['client_dataset_name']]
            if task_type == 'cls_lm':
                data = mta(result, 'fed_moe')
            else:
                data = mtr(result, 'fed_moe')
            metric = data[-1] if last else np.max(data)
            metrics[i].append(metric)

    expert_nums = [int(expert_num) for expert_num in expert_nums]
    max_experts = [int(m) for m in max_experts]
    X, Y = np.meshgrid(expert_nums, max_experts)
    Z = np.array(metrics)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=45, azim=-60)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.5)

    ax.set_xlabel('Total Experts', fontsize=24, labelpad=15)
    ax.set_ylabel('Max Experts', fontsize=24, labelpad=15)
    ax.set_zlabel('Accuracy (%)' if task_type == 'cls_lm' else 'ROUGE (%)', fontsize=24, labelpad=20)
    ax.set_xticks(expert_nums)
    ax.set_yticks(max_experts)
    ax.tick_params(labelsize=24)
    ax.tick_params(axis='z', pad=10)
    cbar = fig.colorbar(surf, ax=ax, pad=0.2)
    cbar.ax.tick_params(labelsize=24)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if save_dir:
        save_path = os.path.join(save_dir, "expert_num_vs_max_experts.pdf")
        fig.savefig(save_path, backend='pgf')

    plt.show()


def plot_logits_scatter(root_dir: str, seed: int, save_dir=None, layer_step=1, sep_legend=False):
    with open(os.path.join(root_dir, 'params.json'), 'r') as f:
        params = json.load(f)
        dataset = params['client_dataset_name']
        client_num = params['client_num']
        rounds = params['rounds']
        lora_rank = params['lora_rank']

    embs_root = os.path.join(root_dir, 'embs')
    legend_elements = []
    for r in [1, rounds - 2]:
        token_embs = {}
        expert_embs = {}
        expert_ids = {}
        expert_nums = {}
        for client_id in range(client_num):
            token_embs_path = os.path.join(embs_root, f'round{r}_client{client_id}_token_embs.pt')
            token_embs_tmp = torch.load(token_embs_path, map_location='cpu')
            expert_embs_path = os.path.join(embs_root, f'round{r}_client{client_id}_expert_embs.pt')
            expert_embs_tmp = torch.load(expert_embs_path, map_location='cpu')
            expert_ids_path = os.path.join(embs_root, f'round{r}_client{client_id}_expert_ids.pkl')
            with open(expert_ids_path, 'rb') as f:
                expert_ids_tmp = pickle.load(f)
            expert_nums_path = os.path.join(embs_root, f'round{r + 1}_client{client_id}_expert_ids.pkl')
            with open(expert_nums_path, 'rb') as f:
                expert_nums_tmp = {k: len(v) for k, v in pickle.load(f).items()}

            for module_name in token_embs_tmp.keys():
                if module_name not in token_embs:
                    token_embs[module_name] = [token_embs_tmp[module_name]]
                    expert_embs[module_name] = [expert_embs_tmp[module_name].mean(dim=0)]
                    expert_ids[module_name] = expert_ids_tmp[module_name]
                    expert_nums[module_name] = [expert_nums_tmp[module_name]]
                else:
                    token_embs[module_name].append(token_embs_tmp[module_name])
                    expert_embs[module_name].append(expert_embs_tmp[module_name].mean(dim=0))
                    expert_ids[module_name].extend(expert_ids_tmp[module_name])
                    expert_nums[module_name].append(expert_nums_tmp[module_name])

        logits = {}
        for module_name in expert_embs.keys():
            expert_embs_tmp = expert_embs[module_name]
            expert_ids_tmp = expert_ids[module_name]

            expert_embs_tmp = torch.cat(expert_embs_tmp, dim=0)  # (num_experts * expert_choices, emb_dim)
            expert_ids_tmp = np.array(expert_ids_tmp)  # (num_experts * expert_choices,)
            order = np.argsort(expert_ids_tmp)
            split_points = np.where(np.diff(expert_ids_tmp[order]) != 0)[0] + 1
            order_parts = np.split(order, split_points)
            for order_part in order_parts:
                expert_embs_tmp[order_part[0], :] = expert_embs_tmp[order_part.tolist(), :].mean(dim=0)

            avg_idx = [order_part.tolist()[0] for order_part in order_parts]
            expert_embs_tmp = expert_embs_tmp[avg_idx, :]  # (num_experts, emb_dim)
            for token_embs_tmp in token_embs[module_name]:
                logits_tmp = torch.matmul(
                    token_embs_tmp,  # (num_batches, emb_dim)
                    expert_embs_tmp.T  # (emb_dim, num_experts)
                ) / math.sqrt(lora_rank)  # (num_batches, num_experts)
                if module_name not in logits:
                    logits[module_name] = [logits_tmp]
                else:
                    logits[module_name].append(logits_tmp)

        pattern = re.compile(r'\.(\d+)\.')
        layer_ids = [int(pattern.search(module_name).group(1)) for module_name in logits.keys()]
        layer_num = max(layer_ids) + 1
        for layer_id in range(0, layer_num, layer_step):
            logits_cur_layer = {k: v for k, v in logits.items() if f'.{layer_id}.' in k}
            for module_id, module_name in enumerate(logits_cur_layer.keys()):
                num_logits = [logits.shape[0] for logits in logits_cur_layer[module_name]]
                split_points = np.cumsum(num_logits)[:-1]
                logits_tmp = torch.cat(logits_cur_layer[module_name], dim=0).float().numpy()
                points = TSNE(n_components=2, random_state=seed).fit_transform(logits_tmp)
                points_parts = np.split(points, split_points)

                fig = plt.figure(figsize=(10, 5))
                ax = fig.subplots(1, 2)
                for client_id, points_part in enumerate(points_parts):
                    color = COLORS[client_id % len(COLORS)]
                    marker = MARKERS[client_id % len(MARKERS)]
                    ax[0].scatter(points_part[:, 0], points_part[:, 1], s=15, linewidth=0,
                                  label=f'Client {client_id + 1}', color=color, marker=marker)

                    if r == 1 and layer_id == 0 and module_id == 0:  # only plot legend once
                        legend_elements.append(plt.scatter([0], [0], s=400, linewidth=0,
                                                           label=f'Client {client_id + 1}',
                                                           color=color, marker=marker))

                if sep_legend and r == 1 and layer_id == 0 and module_id == 0:  # only plot legend once
                    plot_legend(legend_elements, save_dir, "logits_scatter_legend.pdf")

                ax[0].set_xticks([])
                ax[0].set_yticks([])
                if not sep_legend:
                    ax[0].legend(fontsize=10, ncol=2, markerscale=2)

                client_ids = list(range(1, client_num + 1))
                expert_nums_tmp = expert_nums[module_name]
                ax[1].bar(client_ids, expert_nums_tmp, color='#A3D1F8', edgecolor='black', linewidth=0.5)
                ax[1].set_xticks(client_ids)
                ax[1].tick_params(labelsize=20)
                ax[1].set_xlabel('Client ID', fontsize=24)
                ax[1].set_ylabel('Number of Experts', fontsize=24)

                fig.tight_layout()
                fig.show()

                module_name = module_name.split('.')[-1]
                if save_dir is not None:
                    save_path = os.path.join(save_dir, f"{dataset}_round{r}_layer{layer_id}_{module_name}.pdf")
                    fig.savefig(save_path, backend='pgf')


if __name__ == '__main__':
    zh_font = "Songti SC" if platform.system() == 'Darwin' else "SimSun"
    config = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", zh_font],  # Times New Roman for en，SimSun for zh
        "font.size": 14,
        "font.weight": "normal",
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",  # a font similar to Times New Roman, but with math symbols
    }
    plt.rcParams.update(config)

    # Uncomment one of the following parts to generate the corresponding figure in the paper

    # Fig. 5
    root_dir = './logs'
    # save_dir = './figures'
    save_dir = None
    plot_metric_per_round(root_dir, save_dir, n_col=1, last=True, round_step=2, sep_legend=True)

    # Fig. 6
    # root_dir = './logs'
    # save_dir = './figures'
    # plot_metric_per_client_num(root_dir, save_dir, include_ft=True, last=True, sep_legend=True)

    # Fig. 7
    # root_dir = './logs'
    # save_dir = './figures'
    # plot_metric_per_iid(root_dir, save_dir, last=True, sep_legend=True)

    # Fig. 8
    # seed = 42
    # set_seed(seed)
    # save_dir = './figures/logits_scatter'
    # root_dir = "./logs/snli/dir1.0/fed_moe/42/1732286999"
    # plot_logits_scatter(root_dir, seed=seed, save_dir=save_dir, layer_step=3, sep_legend=True)
    # root_dir = "./logs/natural-instruct/meta1/fed_moe/42/1732238685"
    # plot_logits_scatter(root_dir, seed=seed, save_dir=save_dir, layer_step=3, sep_legend=True)

    # Fig. 9
    # root_dir = './logs'
    # save_dir = './figures'
    # plot_expert_choices_vs_lr(root_dir, save_dir, last=True)
    # plot_expert_num_vs_max_experts(root_dir, save_dir, last=True)
