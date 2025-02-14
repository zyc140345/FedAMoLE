from tqdm import tqdm
import torch
import torch.nn as nn
from peft import get_peft_model_state_dict
from baselines.node import Client
from model.loss import CausalLMLoss

Softmax = nn.Softmax(dim=-1)
LogSoftmax = nn.LogSoftmax(dim=-1)


def init_optimizer(model, lr, args) -> torch.optim.Optimizer:
    if args.algorithm == 'fdlora':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr, eps=1e-6 if args.precision == 'fp16' else 1e-8
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr, eps=1e-6 if args.precision == 'fp16' else 1e-8
        )
    return optimizer


def learned_adaptive_mutual(client: Client, recorder, cur_round, args):  # FedAPEN
    client.private_model = client.private_model.to(client.device)
    client.private_model.train()
    client.shared_model = client.shared_model.to(client.device)
    client.shared_model.train()
    lam = client.lam.detach()

    print('Node {0}, ensemble training, training lam: {1}'.format(
        client.idx, torch.mean(lam).item()
    ))
    description = 'Node{:d}: Local Epoch {:d}, loss_private={:.4f} loss_shared={:.4f} loss_ensemble={:.4f}'

    total_loss_private = 0.0
    total_loss_shared = 0.0
    total_loss_ensemble = 0.0
    num_trained = 0
    for epoch in range(client.args.local_epochs):
        lr = client.lr()
        ce_loss = CausalLMLoss()
        private_optimizer = init_optimizer(client.private_model, lr, client.args)
        shared_optimizer = init_optimizer(client.shared_model, lr, client.args)

        if args.client_step is not None:
            num_batches = args.client_step
        else:
            num_batches = len(client.train_loader)
        bar_epoch = tqdm(range(num_batches))

        for _ in bar_epoch:
            private_optimizer.zero_grad()
            shared_optimizer.zero_grad()

            try:
                batch = next(client.train_iter)
            except StopIteration:
                client.train_iter = iter(client.train_loader)
                batch = next(client.train_iter)

            data = {k: v.to(client.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            target = batch['labels'].to(client.device)
            output_private = client.private_model(**data).logits
            output_shared = client.shared_model(**data).logits
            if args.algorithm in ["fed_prompt", "fed_ptuning"]:
                num_virtual_tokens = client.shared_model.active_peft_config.num_virtual_tokens
                output_private = output_private[:, num_virtual_tokens:, :]
                output_shared = output_shared[:, num_virtual_tokens:, :]

            ensemble_output_for_private = lam * output_private + (1 - lam) * output_shared.detach()
            ensemble_output_for_shared = lam * output_private.detach() + (1 - lam) * output_shared
            ensemble_output = lam * output_private.detach() + (1 - lam) * output_shared.detach()

            kl_private = client.kl_loss(LogSoftmax(output_private), Softmax(output_shared.detach()))
            kl_shared = client.kl_loss(LogSoftmax(output_shared), Softmax(output_private.detach()))

            ce_loss_private = ce_loss(output_private, target)
            ce_loss_shared = ce_loss(output_shared, target)
            loss_private = ce_loss_private + kl_private + ce_loss(ensemble_output_for_private, target)
            loss_shared = ce_loss_shared + kl_shared + ce_loss(ensemble_output_for_shared, target)
            loss_ensemble = ce_loss(ensemble_output, target)

            loss_private.backward()
            loss_shared.backward()
            private_optimizer.step()
            shared_optimizer.step()

            total_loss_private += loss_private.detach().cpu().item()
            total_loss_shared += loss_shared.detach().cpu().item()
            total_loss_ensemble += loss_ensemble.cpu().item()
            num_trained += len(batch['input_ids'])

            bar_epoch.set_description(description.format(
                client.idx, epoch + 1,
                total_loss_private / num_trained,
                total_loss_shared / num_trained,
                total_loss_ensemble / num_trained
            ))

    result = {
        "idx": client.idx,
        "loss_private": total_loss_private / num_trained,
        "loss_shared": total_loss_shared / num_trained,
        "loss_ensemble": total_loss_ensemble / num_trained
    }
    return result


def mutual(client: Client, recorder, cur_round, args):  # FedMutual
    client.private_model = client.private_model.to(client.device)
    client.private_model.train()
    client.shared_model = client.shared_model.to(client.device)
    client.shared_model.train()

    description = 'Node{:d}: Local Epoch {:d}, loss_private={:.4f} loss_shared={:.4f}'

    total_loss_private = 0.0
    total_loss_shared = 0.0
    num_trained = 0
    for epoch in range(client.args.local_epochs):
        lr = client.lr()
        ce_loss = CausalLMLoss()
        private_optimizer = init_optimizer(client.private_model, lr, client.args)
        shared_optimizer = init_optimizer(client.shared_model, lr, client.args)

        if args.client_step is not None:
            num_batches = args.client_step
        else:
            num_batches = len(client.train_loader)
        bar_epoch = tqdm(range(num_batches))

        for _ in bar_epoch:
            private_optimizer.zero_grad()
            shared_optimizer.zero_grad()

            try:
                batch = next(client.train_iter)
            except StopIteration:
                client.train_iter = iter(client.train_loader)
                batch = next(client.train_iter)

            data = {k: v.to(client.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            target = batch['labels'].to(client.device)
            output_private = client.private_model(**data).logits
            output_shared = client.shared_model(**data).logits
            if args.algorithm in ["fed_prompt", "fed_ptuning"]:
                num_virtual_tokens = client.shared_model.active_peft_config.num_virtual_tokens
                output_private = output_private[:, num_virtual_tokens:, :]
                output_shared = output_shared[:, num_virtual_tokens:, :]

            kl_private = client.kl_loss(LogSoftmax(output_private), Softmax(output_shared.detach()))
            kl_shared = client.kl_loss(LogSoftmax(output_shared), Softmax(output_private.detach()))

            ce_loss_private = ce_loss(output_private, target)
            ce_loss_shared = ce_loss(output_shared, target)
            loss_private = ce_loss_private + kl_private
            loss_shared = ce_loss_shared + kl_shared

            loss_private.backward()
            loss_shared.backward()
            private_optimizer.step()
            shared_optimizer.step()

            total_loss_private += loss_private.detach().cpu().item()
            total_loss_shared += loss_shared.detach().cpu().item()
            num_trained += len(batch['input_ids'])

            bar_epoch.set_description(description.format(
                client.idx, epoch + 1,
                total_loss_private / num_trained,
                total_loss_shared / num_trained
            ))

    result = {
        "idx": client.idx,
        "loss_private": total_loss_private / num_trained,
        "loss_shared": total_loss_shared / num_trained
    }
    return result


def fed_avg(client: Client, recorder, cur_round, args):  # FedIT (FT), FedPrompt (FT), FedPTuning (FT), FDLoRA
    client.shared_model = client.shared_model.to(client.device)
    client.shared_model.train()

    description = 'Node{:d}: Local Epoch {:d}, loss_shared={:.4f}'

    total_loss_shared = 0.0
    num_trained = 0
    for epoch in range(client.args.local_epochs):
        shared_optimizer = init_optimizer(client.shared_model, client.lr(), client.args)

        if client.args.client_step is not None:
            num_batches = client.args.client_step
        else:
            num_batches = len(client.train_loader)
        bar_epoch = tqdm(range(num_batches))

        for _ in bar_epoch:
            shared_optimizer.zero_grad()

            try:
                batch = next(client.train_iter)
            except StopIteration:
                client.train_iter = iter(client.train_loader)
                batch = next(client.train_iter)

            data = {
                k: v.to(client.device) for k, v in batch.items()
                if k in ['input_ids', 'labels', 'attention_mask']
            }
            loss_shared = client.shared_model(**data).loss

            loss_shared.backward()
            shared_optimizer.step()

            total_loss_shared += loss_shared.detach().cpu().item()  # must detach to avoid memory leak
            num_trained += len(batch['input_ids'])

            bar_epoch.set_description(description.format(
                client.idx, epoch + 1,
                total_loss_shared / num_trained
            ))

    if args.algorithm == 'fdlora' and cur_round % args.sync_freq == 0:  # FDLoRA synchronization
        print("Synchronize private adapter to shared adapter")
        client.private_adapter = {
            k: v.detach().cpu() for k, v in get_peft_model_state_dict(
                client.shared_model, adapter_name='adapter', save_embedding_layers=False
            ).items()
        }

    result = {
        "idx": client.idx,
        "loss_shared": total_loss_shared / num_trained
    }
    return result


class Trainer:
    def __init__(self, recorder, args):
        self.args = args
        self.recorder = recorder
        if args.algorithm.lower() == 'mutual':
            self.train = mutual
        elif args.algorithm.lower() == 'learned_adaptive_training':
            self.train = learned_adaptive_mutual
        elif args.algorithm.lower() in ['fed_avg', 'fed_prompt', 'fed_ptuning', 'fdlora']:
            self.train = fed_avg
        else:
            raise ValueError(f"Unknown algorithm: {args.algorithm}")

    def __call__(self, client, cur_round):
        record = self.train(client, self.recorder, cur_round, self.args)
        if self.recorder is not None:
            self.recorder.add_train_record(record)
