import os
import torch
import numpy as np
import copy
import peft.tuners.lora as lora
import math
import torch.nn.functional as F
import pickle

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizer
from typing import List
from tqdm import tqdm
from model.loss import get_loss_fn
from evaluation import rouge_score
from model.common import get_model, sync_model_with_tokenizer
from model.mole.util import (
    make_peft_params_as_trainable,
    get_experts,
    get_token_proj,
    inject_experts,
    inject_token_proj,
    convert_to_mole,
    TempAccumulator
)
from util import get_device, get_total_size, DescUpdater, ExponentialLR
from optimize import optimize_expert_dispatch


class Server:
    def __init__(self, tokenizer: PreTrainedTokenizer, writer: SummaryWriter, args):
        self.writer = writer
        self.lr = ExponentialLR(1e-3, args.lr_decay)

        self.global_model = get_model(args.model_name, args)
        convert_to_mole(self.global_model, args)
        sync_model_with_tokenizer(self.global_model, tokenizer)
        self.loss_fn = get_loss_fn(args.task_type)
        self.lora_modules = [
            module_name for module_name, module in self.global_model.named_modules()
            if isinstance(module, lora.Linear)
        ]

        self.expert_ids = {
            module_name: list(range(args.expert_num + 1))  # the shared expert is the last one
            for module_name in self.lora_modules
        }
        self.experts = get_experts(self.global_model)
        self.token_proj = get_token_proj(self.global_model)
        self.expert_embs = {}
        self.token_embs = {}
        self.expert_dispatch = {}

        self.clients = []
        self.device = get_device(args)
        self.args = args

    def register_clients(self, clients: List["Client"]):
        self.clients = clients
        for client in self.clients:
            client.server = self
            client.model = self.global_model

    def aggregate(self):
        if self.args.do_profile:
            for client in self.clients:
                client.upload_volume.append(0)

        for module_name in self.lora_modules:
            expert_ids = []
            experts = []
            expert_embs = []
            token_embs = []
            token_projections = []
            do_profile = self.args.do_profile
            for client_idx, client in enumerate(self.clients):
                def profile_and_collect(attr_name, collection, reset_value=None, extend=False):
                    temp = getattr(client, attr_name)[module_name]
                    if do_profile:
                        client.upload_volume[-1] += get_total_size(temp)
                    collection.extend(temp) if extend else collection.append(temp)
                    getattr(client, attr_name)[module_name] = reset_value

                profile_and_collect("expert_ids", expert_ids, [], True)
                profile_and_collect("experts", experts, [], True)
                profile_and_collect("token_proj", token_projections, {})
                if not self.args.static_arch:
                    profile_and_collect("expert_embs", expert_embs, None)
                    profile_and_collect("token_embs", token_embs, None)
            if not self.args.static_arch:
                expert_embs = torch.cat(expert_embs, dim=0)  # (num_experts, r)
                token_embs = torch.cat(token_embs, dim=0)  # (num_clients, r)

            # aggregate the token projection
            token_proj_avg = copy.deepcopy(token_projections[0])
            for key in token_proj_avg.keys():
                for token_proj in token_projections[1:]:
                    token_proj_avg[key] += token_proj[key]
                token_proj_avg[key] = torch.div(token_proj_avg[key], len(self.clients))

            # aggregate the experts with the same IDs
            expert_ids = np.array(expert_ids)
            order = np.argsort(expert_ids)
            split_points = np.where(np.diff(expert_ids[order]) != 0)[0] + 1
            order_parts = np.split(order, split_points)
            for order_part in order_parts:
                expert_weights = [experts[idx] for idx in order_part]
                expert_weight_avg = copy.deepcopy(expert_weights[0])
                for key in expert_weight_avg.keys():
                    for weight in expert_weights[1:]:
                        expert_weight_avg[key] += weight[key]
                    expert_weight_avg[key] = torch.div(expert_weight_avg[key], len(order_part))
                experts[order_part[0]] = expert_weight_avg

            # aggregate the expert embeddings with the same IDs
            if not self.args.static_arch:
                for order_part in order_parts:
                    expert_embs[order_part[0], :] = expert_embs[order_part.tolist(), :].mean(dim=0)

            # save the aggregated results
            self.token_proj[module_name] = token_proj_avg
            avg_idx = [order_part.tolist()[0] for order_part in order_parts]
            self.expert_ids[module_name] = [expert_ids[idx] for idx in avg_idx]
            self.experts[module_name] = [experts[idx] for idx in avg_idx]
            if not self.args.static_arch:
                expert_embs = expert_embs[avg_idx[:-1], :]  # exclude the shared expert
                self.expert_embs[module_name] = expert_embs
                self.token_embs[module_name] = token_embs

    def dispatch_experts(self, r):
        if self.args.do_profile:
            for client in self.clients:
                client.download_volume.append(0)

        with tqdm(self.lora_modules, desc="Dispatching experts") as progress_bar:
            for module_name in progress_bar:
                num_experts = self.args.expert_num  # exclude the shared expert
                num_clients = self.args.client_num
                if r == 0 or not self.args.static_arch:  # construct static architectures at the first round
                    if r == 0 or self.args.random_dispatch:  # randomly dispatch experts at the first round
                        routing_probs = torch.rand(num_experts, num_clients)  # (num_experts, num_clients)
                    else:
                        expert_embs = self.expert_embs[module_name]  # (num_experts, r)
                        token_embs = self.token_embs[module_name]  # (num_clients, r)
                        scale = math.sqrt(self.args.lora_rank)
                        routing_probs = expert_embs @ token_embs.T / scale  # (num_experts, num_clients)
                    routing_probs = F.softmax(routing_probs, dim=1).float().numpy()
                    expert_dispatch = optimize_expert_dispatch(routing_probs, self.args)
                    expert_dispatch = np.vstack((
                        expert_dispatch,
                        np.ones((1, num_clients))  # dispatch the shared expert to all clients
                    ))
                    self.expert_dispatch[module_name] = expert_dispatch

                for client_idx, client in enumerate(self.clients):
                    expert_ids = np.where(self.expert_dispatch[module_name][:, client_idx])[0].tolist()
                    if isinstance(expert_ids, int):
                        expert_ids = [expert_ids]
                    experts = [
                        copy.deepcopy(self.experts[module_name][expert_id])
                        for expert_id in expert_ids
                    ]
                    token_proj = copy.deepcopy(self.token_proj[module_name])

                    if self.args.do_profile:
                        client.download_volume[-1] += get_total_size(expert_ids)
                        client.download_volume[-1] += get_total_size(experts)
                        client.download_volume[-1] += get_total_size(token_proj)

                    client.expert_ids[module_name] = expert_ids
                    client.experts[module_name] = experts
                    client.token_proj[module_name] = token_proj

                # to save memory
                self.experts[module_name] = []
                self.expert_ids[module_name] = []
                self.token_embs[module_name] = None
                self.expert_embs[module_name] = None
                self.token_proj[module_name] = {}


class Client:
    def __init__(
        self, idx: int,
        train_loader: DataLoader, aux_loader: DataLoader,
        eval_loader: DataLoader, test_loader: DataLoader,
        tokenizer: PreTrainedTokenizer, writer: SummaryWriter, args
    ):
        self.idx = idx

        self.train_loader = train_loader
        self.aux_loader = aux_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.train_iter = iter(train_loader)
        self.tokenizer = tokenizer
        self.writer = writer
        self.lr = ExponentialLR(args.lr, args.lr_decay)

        self.model = None
        self.experts = {}
        self.expert_ids = {}
        self.token_proj = {}
        self.token_embs = {}
        self.expert_embs = {}
        self.loss_fn = get_loss_fn(args.task_type)

        self.server = None
        self.device = get_device(args)
        self.args = args
        if self.args.do_profile:
            self.upload_volume = []
            self.download_volume = []

    def load_model(self):
        inject_experts(self.model, self.experts, self.args)
        self.experts = None
        inject_token_proj(self.model, self.token_proj)
        self.token_proj = None

    def release_model(self):
        self.experts = get_experts(self.model)
        self.token_proj = get_token_proj(self.model)

    def compute_embs(self, r):
        accumulator = TempAccumulator(self.model, self.args.save_embs)
        self.model.eval()
        self.model.to(self.device)

        with tqdm(self.aux_loader, desc=f"Client {self.idx} Computing Embeddings") as progress_bar:
            with torch.inference_mode():
                for batch in progress_bar:
                    accumulator.zero_loss()
                    x = {
                        k: v.to(self.device) for k, v in batch.items()
                        if k in ["input_ids", "labels", "attention_mask"]
                    }
                    self.model(**x)

        if self.args.save_embs:
            token_embs_path = os.path.join(self.args.log_root, "embs", f"round{r}_client{self.idx}_token_embs.pt")
            with open(token_embs_path, 'wb') as f:
                torch.save(accumulator.token_embs, f)
            expert_embs_path = os.path.join(self.args.log_root, "embs", f"round{r}_client{self.idx}_expert_embs.pt")
            with open(expert_embs_path, 'wb') as f:
                torch.save(accumulator.expert_embs, f)
            expert_ids_path = os.path.join(self.args.log_root, "embs", f"round{r}_client{self.idx}_expert_ids.pkl")
            with open(expert_ids_path, 'wb') as f:
                expert_ids = {
                    module_name: expert_ids[:-1]  # exclude the shared expert
                    for module_name, expert_ids in self.expert_ids.items()
                }
                pickle.dump(expert_ids, f)

        self.expert_embs = accumulator.mean_expert_embs
        self.token_embs = accumulator.mean_token_embs
        accumulator.unregister()

    def train_experts(self, r: int):
        make_peft_params_as_trainable(self.model)
        accumulator = TempAccumulator(self.model)
        self.model.train()
        self.model.to(self.device)
        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr(), eps=1e-6 if self.args.precision == 'fp16' else 1e-8
        )

        num_epochs = self.args.local_epochs
        for e in range(num_epochs):
            num_trained = 0
            total_loss = 0
            if self.args.client_step is not None:
                num_batches = self.args.client_step
            else:
                num_batches = len(self.train_loader)
            progress_bar = tqdm(range(num_batches))
            desc_updater = DescUpdater(
                progress_bar, self.args.task_type,
                prefix=f"Client {self.idx} " + "Train Epoch {}"
            )
            for batch_idx in progress_bar:
                accumulator.zero_loss()
                optimizer.zero_grad()

                try:
                    batch = next(self.train_iter)
                except StopIteration:
                    self.train_iter = iter(self.train_loader)
                    batch = next(self.train_iter)

                x = {
                    k: v.to(self.device) for k, v in batch.items()
                    if k in ["input_ids", "labels", "attention_mask"]
                }
                loss = self.model(**x).loss
                total_loss += loss.item()
                loss += self.args.load_balance_alpha * accumulator.load_balance_loss

                loss.backward()
                optimizer.step()

                num_trained += len(x["input_ids"])
                avg_loss = total_loss / num_trained
                desc_updater.update(e, avg_loss)

                global_step = (r - 1) * num_epochs * num_batches + (r - 1) * e * num_batches + batch_idx
                self.writer.add_scalar(f"Loss/Train Experts/Client {self.idx}", avg_loss, global_step)

        accumulator.unregister()

    def eval(self):
        self.model.eval()
        self.model.to(self.device)

        total_loss = 0
        num_evaluated = 0
        avg_loss = 0
        with tqdm(self.eval_loader) as progress_bar:
            desc_updater = DescUpdater(progress_bar, self.args.task_type, prefix="Eval")
            with torch.inference_mode():
                for batch in progress_bar:
                    x = {
                        k: v.to(self.device) for k, v in batch.items()
                        if k in ["input_ids", "labels", "attention_mask"]
                    }
                    loss = self.model(**x).loss

                    total_loss += loss.item()
                    num_evaluated += len(x["input_ids"])
                    avg_loss = total_loss / num_evaluated
                    desc_updater.update(avg_loss)

        return avg_loss

    def test(self, r: int):
        self.model.eval()
        self.model.to(self.device)

        total_loss = 0
        total_correct = 0
        num_tested = 0
        avg_loss = 0
        avg_acc = 0
        with tqdm(self.test_loader) as progress_bar:
            desc_updater = DescUpdater(progress_bar, self.args.task_type, train=False, prefix="Test")
            with torch.inference_mode():
                for batch in progress_bar:
                    x = {k: v.to(self.device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
                    if self.args.model_name == 'switch-base-32':
                        x['labels'] = batch['labels'].to(self.device)
                    y = batch['labels'].to(self.device)
                    if self.args.task_type == 'causal_lm':
                        y_hat = self.model.generate(**x, max_new_tokens=128, num_beams=1)
                        y_hat = [y_hat[i, x["input_ids"].shape[-1]:] for i in range(y_hat.shape[0])]
                        total_loss += rouge_score(y_hat, y, self.tokenizer)
                    else:
                        truth = batch['class'].to(self.device)
                        y_hat = self.model(**{k: v.view(-1, v.shape[-1]) for k, v in x.items()}).logits
                        loss, loss_all = self.loss_fn(y_hat, y, truth)
                        total_loss += loss.item()

                    num_tested += len(x["input_ids"])
                    avg_loss = total_loss / num_tested
                    if self.args.task_type == "causal_lm":
                        desc_updater.update(avg_loss)
                    else:
                        pred = torch.stack([torch.argmin(loss) for loss in loss_all.detach()], dim=0)
                        total_correct += (pred == truth).sum().item()
                        avg_acc = total_correct / num_tested
                        desc_updater.update(avg_loss, avg_acc * 100)

        self.writer.add_scalar(f"Loss/Test/Client {self.idx}", avg_loss, r)
        if self.args.task_type == "cls_lm":
            self.writer.add_scalar(f"Acc/Test/Client {self.idx}", avg_acc * 100, r)

        return avg_loss, avg_acc
