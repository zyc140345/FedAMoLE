import torch

from tqdm import tqdm
from baselines.node import Client
from evaluation import rouge_score
from baselines.ensemble_model import EnsembleCausalLMModel
from torch import nn
from model.loss import CausalLMLoss

Softmax = nn.Softmax(dim=-1)
LogSoftmax = nn.LogSoftmax(dim=-1)


class Evaluator:
    def __init__(self, args):
        self.args = args

        self.val_loss = [[] for _ in range(args.client_num)]
        self.val_loss_mutual_private = [[] for _ in range(args.client_num)]
        self.val_loss_mutual_shared = [[] for _ in range(args.client_num)]
        self.val_loss_mutual_ensemble = [[] for _ in range(args.client_num)]
        self.val_loss_mutual_ensemble_adaptive = [[] for _ in range(args.client_num)]

        self.test_loss = [[] for _ in range(args.client_num)]
        self.test_loss_mutual_private = [[] for _ in range(args.client_num)]
        self.test_loss_mutual_shared = [[] for _ in range(args.client_num)]
        self.test_loss_mutual_ensemble = [[] for _ in range(args.client_num)]
        self.test_loss_mutual_ensemble_adaptive = [[] for _ in range(args.client_num)]
        if args.task_type == 'cls_lm':
            self.test_acc = [[] for _ in range(args.client_num)]
            self.test_acc_mutual_private = [[] for _ in range(args.client_num)]
            self.test_acc_mutual_shared = [[] for _ in range(args.client_num)]
            self.test_acc_mutual_ensemble = [[] for _ in range(args.client_num)]
            self.test_acc_mutual_ensemble_adaptive = [[] for _ in range(args.client_num)]

    def eval_single_model(self, client):
        client.shared_model = client.shared_model.to(client.device)
        client.shared_model.eval()

        total_loss = 0.0
        num_eval = 0

        with torch.inference_mode():
            description = 'Node{:d} Eval: loss_shared={:.4f}'
            with tqdm(client.eval_loader) as bar_eval:
                for idx, batch in enumerate(bar_eval):
                    data = {
                        k: v.to(client.device) for k, v in batch.items()
                        if k in ['input_ids', 'labels', 'attention_mask']
                    }
                    loss = client.shared_model(**data).loss

                    total_loss += loss
                    num_eval += len(data["input_ids"])

                    bar_eval.set_description(description.format(
                        client.idx,
                        total_loss / num_eval
                    ))

        self.val_loss[client.idx].append((total_loss / num_eval).item())

    def eval_ensemble_model(self, client):
        client.private_model = client.private_model.to(client.device)
        client.private_model.eval()
        client.shared_model = client.shared_model.to(client.device)
        client.shared_model.eval()

        if client.args.algorithm in ['learned_adaptive_training', 'mutual']:
            lam = client.lam.detach()

        total_loss_private = 0.0
        total_loss_shared = 0.0
        total_loss_ensemble = 0.0
        total_loss_ensemble_adaptive = 0.0
        num_eval = 0

        with torch.inference_mode():
            if client.args.algorithm == 'learned_adaptive_training':
                ensemble_metric_suffix = 'ensemble_adaptive'
            else:
                ensemble_metric_suffix = 'ensemble'
            description = 'Node{:d} Eval: loss_private={:.4f} loss_shared={:.4f} loss_{}={:.4f}'

            eval_loader = client.eval_loader
            ce_loss = CausalLMLoss()
            with tqdm(eval_loader) as bar_eval:
                for idx, batch in enumerate(bar_eval):
                    data = {k: v.to(client.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
                    target = batch['labels'].to(client.device)
                    output_private: torch.Tensor = client.private_model(**data).logits
                    output_shared: torch.Tensor = client.shared_model(**data).logits

                    output_ensemble = (output_private + output_shared) / 2
                    output_ensemble_adaptive = lam * output_private + (1 - lam) * output_shared

                    ce_loss_private = ce_loss(output_private, target)
                    ce_loss_shared = ce_loss(output_shared, target)
                    ce_loss_ensemble = ce_loss(output_ensemble, target)
                    ce_loss_ensemble_adaptive = ce_loss(output_ensemble_adaptive, target)

                    total_loss_private += ce_loss_private
                    total_loss_shared += ce_loss_shared
                    total_loss_ensemble += ce_loss_ensemble
                    total_loss_ensemble_adaptive += ce_loss_ensemble_adaptive
                    num_eval += len(data["input_ids"])

                    ensemble_loss = total_loss_ensemble_adaptive / num_eval
                    bar_eval.set_description(description.format(
                        client.idx,
                        total_loss_private / num_eval,
                        total_loss_shared / num_eval,
                        ensemble_metric_suffix, ensemble_loss
                    ))

            total_loss_private = (total_loss_private / num_eval).item()
            total_loss_shared = (total_loss_shared / num_eval).item()
            total_loss_ensemble = (total_loss_ensemble / num_eval).item()
            total_loss_ensemble_adaptive = (total_loss_ensemble_adaptive / num_eval).item()

        self.val_loss_mutual_private[client.idx].append(total_loss_private)
        self.val_loss_mutual_shared[client.idx].append(total_loss_shared)
        self.val_loss_mutual_ensemble[client.idx].append(total_loss_ensemble)
        self.val_loss_mutual_ensemble_adaptive[client.idx].append(total_loss_ensemble_adaptive)

    def test_single_model(self, client):
        client.shared_model = client.shared_model.to(client.device)
        client.shared_model.eval()
        model = client.shared_model

        total_loss = 0.0
        correct = 0.0
        num_test = 0

        with torch.inference_mode():
            if client.args.task_type == 'cls_lm':
                description = 'Node{:d} Test: loss_shared={:.4f} acc_shared={:.4f}%'
            else:
                description = 'Node{:d} Test: loss_shared={:.4f}'

            with tqdm(client.test_loader) as bar_test:
                for idx, batch in enumerate(bar_test):
                    data = {k: v.to(client.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
                    if self.args.task_type == 'cls_lm':
                        target = batch['labels'].to(client.device)
                        classes = batch['class'].to(client.device)
                        if self.args.model_name == 'switch-base-32':
                            data['labels'] = target
                        output = model(**{k: v.view(-1, v.shape[-1]) for k, v in data.items()}).logits
                        if client.args.algorithm in ["fed_prompt", "fed_ptuning"]:
                            num_virtual_tokens = client.shared_model.active_peft_config.num_virtual_tokens
                            output = output[:, num_virtual_tokens:, :]
                        loss, loss_all = client.ce_loss(output, target, classes)
                        total_loss += loss.item()
                    else:
                        target = batch['labels']
                        output = model.generate(**data, max_new_tokens=128, num_beams=1)
                        output = [output[i, data["input_ids"].shape[-1]:] for i in range(output.shape[0])]
                        total_loss += rouge_score(output, target, client.tokenizer)

                    num_test += len(data["input_ids"])
                    if client.args.task_type == 'cls_lm':
                        pred = torch.stack([torch.argmin(loss) for loss in loss_all], dim=0)
                        correct += (pred == classes).sum().item()
                        bar_test.set_description(description.format(
                            client.idx,
                            total_loss / num_test,
                            correct / num_test * 100
                        ))
                    else:
                        bar_test.set_description(description.format(
                            client.idx,
                            total_loss / num_test
                        ))

            total_loss = total_loss / num_test
            if self.args.task_type == 'cls_lm':
                acc = correct / len(client.test_loader.dataset) * 100

        self.test_loss[client.idx].append(total_loss)
        if self.args.task_type == 'cls_lm':
            self.test_acc[client.idx].append(acc)

    def test_ensemble_model(self, client: Client):
        client.private_model = client.private_model.to(client.device)
        client.private_model.eval()
        client.shared_model = client.shared_model.to(client.device)
        client.shared_model.eval()
        ensemble_model = None
        adaptive_ensemble_model = None
        lam_static = 0.5
        lam_ada = 0.5

        if client.args.algorithm == 'mutual':
            print(f'Node {client.idx}, test lam: {lam_static}')
        else:
            lam_ada = client.lam.detach()
            print(f'Node {client.idx}, test lam: {lam_ada.mean().item()}')

        if client.args.task_type == 'causal_lm':
            if client.args.algorithm == 'mutual' or client.args.log_level == 'detailed':
                ensemble_model = EnsembleCausalLMModel(
                    client.private_model, client.shared_model, lam_static
                ).to(client.device)
                ensemble_model.eval()
            if client.args.algorithm == 'learned_adaptive_training':
                adaptive_ensemble_model = EnsembleCausalLMModel(
                    client.private_model, client.shared_model, lam_ada
                ).to(client.device)
                adaptive_ensemble_model.eval()

        total_loss_private = 0.0
        total_loss_shared = 0.0
        total_loss_ensemble = 0.0
        total_loss_ensemble_adaptive = 0.0
        correct_private = 0.0
        correct_shared = 0.0
        correct_ensemble = 0.0
        correct_ensemble_adaptive = 0.0
        num_test = 0

        with torch.inference_mode():
            if client.args.algorithm == 'learned_adaptive_training':
                ensemble_metric_suffix = 'ensemble_adaptive'
            else:
                ensemble_metric_suffix = 'ensemble'

            if client.args.task_type == 'cls_lm':
                if client.args.log_level == 'summarized':
                    description = 'Node{:d} Test: loss_{}={:.4f} acc_{}={:.2f}%'
                else:
                    description = 'Node{:d} Test: loss_private={:.4f} acc_private={:.2f}% ' \
                                  'loss_shared={:.4f} acc_shared={:.2f}% ' \
                                  'loss_{}={:.4f} acc_{}={:.2f}%'
            else:
                if client.args.log_level == 'summarized':
                    description = 'Node{:d} Test: loss_{}={:.4f}'
                else:
                    description = 'Node{:d} Test: loss_private={:.4f} loss_shared={:.4f} loss_{}={:.4f}'

            test_loader = client.test_loader
            with tqdm(test_loader) as bar_test:
                for idx, batch in enumerate(bar_test):
                    data = {k: v.to(client.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
                    if self.args.task_type == 'cls_lm':
                        target = batch['labels'].to(client.device)
                        classes = batch['class'].to(client.device)
                        output_private = client.private_model(**{
                            k: v.view(-1, v.shape[-1]) for k, v in data.items()
                        }).logits
                        output_shared = client.shared_model(**{
                            k: v.view(-1, v.shape[-1]) for k, v in data.items()
                        }).logits
                        output_ensemble = lam_static * output_private + (1 - lam_static) * output_shared
                        output_ensemble_adaptive = lam_ada * output_private + (1 - lam_ada) * output_shared
                    else:
                        target = batch['labels']
                        if self.args.log_level == 'detailed':
                            output_private = client.private_model.generate(
                                **data, max_new_tokens=128, num_beams=1
                            )
                            output_private = [
                                output_private[i, data["input_ids"].shape[-1]:]
                                for i in range(output_private.shape[0])
                            ]
                            output_shared = client.shared_model.generate(
                                **data, max_new_tokens=128, num_beams=1
                            )
                            output_shared = [
                                output_shared[i, data["input_ids"].shape[-1]:]
                                for i in range(output_shared.shape[0])
                            ]
                        if client.args.algorithm == 'mutual' or client.args.log_level == 'detailed':
                            output_ensemble = ensemble_model.generate(
                                **data, max_new_tokens=128, num_beams=1
                            )
                            output_ensemble = [
                                output_ensemble[i, data["input_ids"].shape[-1]:]
                                for i in range(output_ensemble.shape[0])
                            ]
                        if client.args.algorithm == 'learned_adaptive_training':
                            output_ensemble_adaptive = adaptive_ensemble_model.generate(
                                **data, max_new_tokens=128, num_beams=1
                            )
                            output_ensemble_adaptive = [
                                output_ensemble_adaptive[i, data["input_ids"].shape[-1]:]
                                for i in range(output_ensemble_adaptive.shape[0])
                            ]

                    num_test += len(data["input_ids"])
                    if client.args.task_type == 'cls_lm':
                        ce_loss_private, ce_loss_private_all = client.ce_loss(output_private, target, classes)
                        ce_loss_shared, ce_loss_shared_all = client.ce_loss(output_shared, target, classes)
                        ce_loss_ensemble, ce_loss_ensemble_all = client.ce_loss(output_ensemble, target, classes)
                        ce_loss_ensemble_adaptive, ce_loss_ensemble_adaptive_all = client.ce_loss(
                            output_ensemble_adaptive, target, classes
                        )
                        total_loss_private += ce_loss_private.item()
                        total_loss_shared += ce_loss_shared.item()
                        total_loss_ensemble += ce_loss_ensemble.item()
                        total_loss_ensemble_adaptive += ce_loss_ensemble_adaptive.item()

                        pred_private = torch.stack([torch.argmin(loss) for loss in ce_loss_private_all], dim=0)
                        pred_shared = torch.stack([torch.argmin(loss) for loss in ce_loss_shared_all], dim=0)
                        pred_ensemble = torch.stack([torch.argmin(loss) for loss in ce_loss_ensemble_all], dim=0)
                        pred_ensemble_adaptive = torch.stack(
                            [torch.argmin(loss) for loss in ce_loss_ensemble_adaptive_all], dim=0
                        )
                        correct_private += (pred_private == classes).sum().item()
                        correct_shared += (pred_shared == classes).sum().item()
                        correct_ensemble += (pred_ensemble == classes).sum().item()
                        correct_ensemble_adaptive += (pred_ensemble_adaptive == classes).sum().item()

                        if client.args.algorithm == 'mutual':
                            ensemble_loss = total_loss_ensemble / num_test
                            ensemble_acc = correct_ensemble / num_test * 100
                        else:
                            ensemble_loss = total_loss_ensemble_adaptive / num_test
                            ensemble_acc = correct_ensemble_adaptive / num_test * 100

                        if client.args.log_level == 'detailed':
                            bar_test.set_description(description.format(
                                client.idx,
                                total_loss_private / num_test,
                                correct_private / num_test * 100,
                                total_loss_shared / num_test,
                                correct_shared / num_test * 100,
                                ensemble_metric_suffix, ensemble_loss,
                                ensemble_metric_suffix, ensemble_acc
                            ))
                        else:
                            bar_test.set_description(description.format(
                                client.idx,
                                ensemble_metric_suffix, ensemble_loss,
                                ensemble_metric_suffix, ensemble_acc
                            ))
                    else:
                        if client.args.log_level == 'detailed':
                            total_loss_private += rouge_score(output_private, target, client.tokenizer)
                            total_loss_shared += rouge_score(output_shared, target, client.tokenizer)
                        if client.args.algorithm == 'mutual' or client.args.log_level == 'detailed':
                            total_loss_ensemble += rouge_score(output_ensemble, target, client.tokenizer)
                        if client.args.algorithm == 'learned_adaptive_training':
                            total_loss_ensemble_adaptive += rouge_score(
                                output_ensemble_adaptive, target, client.tokenizer
                            )

                        if client.args.algorithm == 'mutual':
                            ensemble_loss = total_loss_ensemble / num_test
                        else:
                            ensemble_loss = total_loss_ensemble_adaptive / num_test

                        if client.args.log_level == 'detailed':
                            bar_test.set_description(description.format(
                                client.idx,
                                total_loss_private / num_test,
                                total_loss_shared / num_test,
                                ensemble_metric_suffix, ensemble_loss
                            ))
                        else:
                            bar_test.set_description(description.format(
                                client.idx,
                                ensemble_metric_suffix, ensemble_loss
                            ))

            total_loss_private = total_loss_private / num_test
            total_loss_shared = total_loss_shared / num_test
            total_loss_ensemble = total_loss_ensemble / num_test
            total_loss_ensemble_adaptive = total_loss_ensemble_adaptive / num_test
            if client.args.task_type == 'cls_lm':
                acc_private = correct_private / len(test_loader) * 100
                acc_shared = correct_shared / len(test_loader) * 100
                acc_ensemble = correct_ensemble / len(test_loader) * 100
                acc_ensemble_adaptive = correct_ensemble_adaptive / len(test_loader) * 100

        self.test_loss_mutual_private[client.idx].append(total_loss_private)
        self.test_loss_mutual_shared[client.idx].append(total_loss_shared)
        self.test_loss_mutual_ensemble[client.idx].append(total_loss_ensemble)
        self.test_loss_mutual_ensemble_adaptive[client.idx].append(total_loss_ensemble_adaptive)
        if client.args.task_type == 'cls_lm':
            self.test_acc_mutual_private[client.idx].append(acc_private)
            self.test_acc_mutual_shared[client.idx].append(acc_shared)
            self.test_acc_mutual_ensemble[client.idx].append(acc_ensemble)
            self.test_acc_mutual_ensemble_adaptive[client.idx].append(acc_ensemble_adaptive)
