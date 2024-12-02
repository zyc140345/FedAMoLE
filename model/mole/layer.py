import torch
import torch.nn.functional as F
import peft.tuners.lora as lora
import math
from typing import Any


class MoLELinear:
    def __init__(self, module_name: str, origin_module: lora.Linear, k: int = 2):
        self.module_name = module_name
        self.origin_module = origin_module
        self.r = list(self.origin_module.r.values())[0]
        self.k = k
        self.accumulator = None

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        previous_dtype = x.dtype
        result = self.origin_module.base_layer(x, *args, **kwargs)

        shared_output = self._forward_expert(x, 1, "expert_shared")
        result += shared_output

        bsz, seq_len, dim = x.size()
        x = x.view(-1, dim)

        num_experts = len(self.origin_module.active_adapter) - 1  # exclude the shared expert
        expert_proj = torch.cat([
            lora_A.weight for expert_name, lora_A in self.origin_module.lora_A.items()
            if expert_name != "expert_shared"
        ], dim=0)  # (num_experts * r, dim)
        expert_embs = F.linear(x, expert_proj).view(x.shape[0], -1, self.r)  # (bsz * seq_len, num_experts, r)
        token_embs = self.origin_module.token_proj(x)  # (bzs * seq_len, r)
        scaled_dot_product = torch.bmm(expert_embs, token_embs.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.r)
        routing_probs = F.softmax(scaled_dot_product, dim=-1)  # (bsz * seq_len, num_experts)
        _, selected_experts = routing_probs.topk(self.k, dim=-1)  # (bsz * seq_len, k)
        num_tokens = F.one_hot(selected_experts.view(-1), num_experts).sum(0)  # (num_experts,)

        if self.accumulator:
            P = routing_probs.mean(dim=0)
            f = (num_tokens / num_tokens.sum()).to(dtype=P.dtype)
            loss = num_experts * torch.dot(P, f)
            self.accumulator.add_load_balance_loss(loss)
            self.accumulator.add_embs(self.module_name, token_embs, expert_embs)

        order = selected_experts.view(-1).argsort(0)
        x = x.unsqueeze(1).repeat(1, self.k, 1).view(-1, dim)[order]  # reorder according to expert number
        x = x.split(num_tokens.tolist(), dim=0)
        routing_probs = routing_probs.gather(dim=1, index=selected_experts)  # (bsz * seq_len, k)
        routing_probs /= routing_probs.sum(dim=1, keepdim=True)  # normalize
        routing_probs = routing_probs.view(-1, 1)[order]
        routing_probs = routing_probs.split(num_tokens.tolist(), dim=0)

        # todo: parallelize
        moe_output = []
        for expert_idx, expert_name in enumerate(self.origin_module.active_adapter[:-1]):  # exclude the shared expert
            moe_output.append(self._forward_expert(x[expert_idx], routing_probs[expert_idx], expert_name))
        moe_output = torch.vstack(moe_output)
        moe_output = moe_output[order.argsort(0)]  # restore original order
        moe_output = moe_output.view(bsz, seq_len, self.k, -1).sum(dim=2)

        result += moe_output
        result = result.to(previous_dtype)
        return result

    def _forward_expert(self, x, routing_prob, expert_name):
        lora_A = self.origin_module.lora_A[expert_name]
        lora_B = self.origin_module.lora_B[expert_name]
        dropout = self.origin_module.lora_dropout[expert_name]
        scaling = self.origin_module.scaling[expert_name]
        x = x.to(lora_A.weight.dtype)
        return lora_B(lora_A(dropout(x))) * scaling * routing_prob
