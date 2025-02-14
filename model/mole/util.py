import torch
import peft.tuners.lora as lora

from torch import nn
from typing import List, Dict
from model.mole.layer import MoLELinear
from util import StateDict
from peft import (
    LoraConfig,
    TaskType,
    PeftMixedModel
)


class TempAccumulator:
    def __init__(self, model: PeftMixedModel, save_embs=False):
        for module_name, module in model.named_modules():
            if isinstance(module, lora.Linear):
                module.new_module.accumulator = self
        self._model = model
        self._save_embs = save_embs

        self.load_balance_loss = 0
        self._num_tokens = {}  # number of tokens per expert
        self._mean_expert_embs = {}
        self._mean_token_embs = {}
        self._expert_embs = {}
        self._token_embs = {}

    def add_embs(self, module_name: str, token_embs: torch.Tensor, expert_embs: torch.Tensor):
        # token_embs: (bsz * seq_len, r)
        # expert_embs: (bsz * seq_len, num_experts, r)

        if module_name not in self._num_tokens:
            self._num_tokens[module_name] = token_embs.shape[0]

            token_embs = token_embs.mean(dim=0).detach()  # (r,)
            self._mean_token_embs[module_name] = token_embs
            if self._save_embs:
                self._token_embs[module_name] = [token_embs.unsqueeze(0)]  # (1, r)

            expert_embs = expert_embs.mean(dim=0).detach()  # (num_experts, r)
            if self._save_embs:
                self._expert_embs[module_name] = [expert_embs.unsqueeze(0)]  # (1, num_experts, r)
            shared_expert_emb = torch.zeros_like(expert_embs[:1, :])  # placeholder for the shared expert
            expert_embs = torch.cat([expert_embs, shared_expert_emb])
            self._mean_expert_embs[module_name] = expert_embs
        else:
            num_tokens = token_embs.shape[0]
            self._num_tokens[module_name] += num_tokens

            if self._save_embs:
                mean_token_embs = token_embs.mean(dim=0, keepdim=True).detach()  # (1, r)
                self._token_embs[module_name].append(mean_token_embs)
            token_embs = token_embs.sum(dim=0).detach()  # (r,)
            self._mean_token_embs[module_name] += \
                (token_embs - num_tokens * self._mean_token_embs[module_name]) / self._num_tokens[module_name]

            if self._save_embs:
                mean_expert_embs = expert_embs.mean(dim=0, keepdim=True).detach()  # (1, num_experts, r)
                self._expert_embs[module_name].append(mean_expert_embs)
            expert_embs = expert_embs.sum(dim=0).detach()  # (num_experts, r)
            # exclude the shared expert
            self._mean_expert_embs[module_name][:-1] += \
                (expert_embs - num_tokens * self._mean_expert_embs[module_name][:-1]) / self._num_tokens[module_name]

    def add_load_balance_loss(self, load_balance_loss: torch.Tensor):
        self.load_balance_loss += load_balance_loss

    @property
    def mean_token_embs(self):
        return {
            module_name: token_emb.unsqueeze(0).cpu()
            for module_name, token_emb in self._mean_token_embs.items()
        }

    @property
    def mean_expert_embs(self):
        return {
            module_name: expert_emb.cpu()
            for module_name, expert_emb in self._mean_expert_embs.items()
        }

    @property
    def token_embs(self):
        return {
            module_name: torch.cat(token_embs, dim=0).cpu()  # (num_batches, r)
            for module_name, token_embs in self._token_embs.items()
        }

    @property
    def expert_embs(self):
        return {
            module_name: torch.cat(expert_embs, dim=0).cpu()  # (num_batches, num_experts, r)
            for module_name, expert_embs in self._expert_embs.items()
        }

    def unregister(self):
        for module in self._model.modules():
            if isinstance(module, lora.Linear):
                module.new_module.accumulator = None

    def zero_loss(self):
        self.load_balance_loss = 0


def convert_to_mole(model: PeftMixedModel, args):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=['q', 'v'] if args.model_name == 'switch-base-32' else None,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    expert_names = []
    for i in range(args.expert_num):
        model.add_adapter(lora_config, adapter_name=f"expert_{i}")
        expert_names.append(f"expert_{i}")
    model.add_adapter(lora_config, adapter_name="expert_shared")
    expert_names.append("expert_shared")
    model.set_adapter(expert_names)
    inject_token_proj(model)

    for module_name, module in model.named_modules():
        if isinstance(module, lora.Linear):
            new_module = MoLELinear(module_name, module, args.top_k)
            module.forward = new_module.forward
            module.new_module = new_module


def inject_experts(model: PeftMixedModel, experts: Dict[str, List[StateDict]], args):
    for module_name, module in model.named_modules():
        if isinstance(module, lora.Linear):
            for expert_name in module.active_adapter:
                module.delete_adapter(expert_name)
            expert_names = []
            for expert in experts[module_name]:
                expert_name = list(expert.keys())[0].split(".")[-2]
                expert_names.append(expert_name)
                module.update_layer(
                    adapter_name=expert_name,
                    r=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    init_lora_weights=True,
                    use_rslora=False
                )
                module.load_state_dict(expert, strict=False)
            module.set_adapter(expert_names)


def inject_token_proj(model: PeftMixedModel, token_proj: Dict[str, StateDict] = None):
    for module_name, module in model.named_modules():
        if isinstance(module, lora.Linear):
            lora_rank = list(module.r.values())[0]
            module.token_proj = nn.Linear(
                in_features=module.in_features,
                out_features=lora_rank,
                bias=False,
                dtype=module.weight.dtype
            )
            if token_proj is not None:
                module.load_state_dict(token_proj[module_name], strict=False)


def get_experts(model: PeftMixedModel) -> Dict[str, List[StateDict]]:
    experts = {}
    for module_name, module in model.named_modules():
        if isinstance(module, lora.Linear):
            experts_cur_module = [{} for _ in range(len(module.active_adapter))]
            for i, expert_name in enumerate(module.active_adapter):
                experts_cur_module[i].update({
                    ".".join(["lora_A", expert_name, k]): v.to('cpu')
                    for k, v in module.lora_A[expert_name].state_dict().items()
                })
                experts_cur_module[i].update({
                    ".".join(["lora_B", expert_name, k]): v.to('cpu')
                    for k, v in module.lora_B[expert_name].state_dict().items()
                })
            experts[module_name] = experts_cur_module
    return experts


def get_token_proj(model: PeftMixedModel) -> Dict[str, StateDict]:
    token_proj = {}
    for module_name, module in model.named_modules():
        if isinstance(module, lora.Linear):
            token_proj[module_name] = {
                ".".join(["token_proj", k]): v.to('cpu')
                for k, v in module.token_proj.state_dict().items()
            }
    return token_proj


def make_peft_params_as_trainable(model: PeftMixedModel):
    for param_name, param in model.named_parameters():
        if 'lora' in param_name or 'token_proj' in param_name:
            param.requires_grad = True
        else:
            param.requires_grad = False
