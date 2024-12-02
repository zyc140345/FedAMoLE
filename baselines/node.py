import torch.optim
import torch.nn as nn
import nevergrad as ng

from copy import deepcopy
from functools import partial
from peft import set_peft_model_state_dict, get_peft_model_state_dict
from model.common import get_model, sync_model_with_tokenizer
from model.loss import ClsLMLoss, CausalLMLoss
from baselines.fdlora import get_score, get_final_adapter, ProgressBar
from util import ExponentialLR, get_total_size


ALGO2PEFT_METHOD = {
    "fed_avg": "lora",
    "learned_adaptive_training": "lora",
    "mutual": "lora",
    "fed_prompt": "prompt_tuning",
    "fed_ptuning": "ptuning",
    "fdlora": "lora"
}


class Server:
    def __init__(self, recorder, tokenizer, args):
        self.args = args

        self.shared_adapter = None
        self.shared_optimizer = None

        self.shared_backbone = get_model(
            args.model_name, self.args,
            no_adapter=False, adapter_name="adapter",
            peft_method=ALGO2PEFT_METHOD[args.algorithm], tokenizer=tokenizer
        )
        sync_model_with_tokenizer(self.shared_backbone, tokenizer)
        if self.args.algorithm in ['learned_adaptive_training', 'mutual']:
            self.private_backbone = get_model(
                args.model_name, self.args,
                no_adapter=False, adapter_name="adapter",
                peft_method=ALGO2PEFT_METHOD[args.algorithm], tokenizer=tokenizer
            )
            sync_model_with_tokenizer(self.private_backbone, tokenizer)

        self.device = torch.device(f'cuda:{args.gpu}')
        self.recorder = recorder
        self.recorder.register_server(self)

    def aggregate(self, clients: list, cur_round):
        if cur_round == -2:
            init_adapter = {
                k: v.detach().cpu() for k, v in get_peft_model_state_dict(
                    self.shared_backbone, adapter_name='adapter', save_embedding_layers=False
                ).items()
            }
            for client in clients:
                client.shared_adapter = deepcopy(init_adapter)
                if self.args.do_profile:
                    client.download_volume.append(get_total_size(client.shared_adapter))
                client.shared_model = self.shared_backbone
                if self.args.algorithm in ['learned_adaptive_training', 'mutual']:
                    client.private_adapter = deepcopy(init_adapter)
                    client.private_model = self.private_backbone
        elif self.args.algorithm == 'fdlora' and cur_round >= 0:  # FDLoRA OuterOpt
            self.shared_optimizer.zero_grad()
            for client in clients:
                weight = client.shared_adapter
                if self.args.do_profile:
                    client.upload_volume.append(get_total_size(weight))
                for key in weight.keys():
                    if self.shared_adapter[key].grad is None:
                        self.shared_adapter[key].grad = self.shared_adapter[key] - weight[key]
                    else:
                        self.shared_adapter[key].grad += self.shared_adapter[key] - weight[key]
            for key in self.shared_adapter.keys():
                self.shared_adapter[key].grad /= len(clients)
            self.shared_optimizer.step()

            for client in clients:
                client.shared_adapter = deepcopy(self.shared_adapter)
                if self.args.do_profile:
                    client.download_volume.append(get_total_size(client.shared_adapter))
        else:
            weights = [client.shared_adapter for client in clients]
            if self.args.do_profile:
                for client in clients:
                    client.upload_volume.append(get_total_size(client.shared_adapter))

            # global aggregation
            weight_avg = deepcopy(weights[0])
            for key in weight_avg.keys():
                for i in range(1, len(weights)):
                    weight_avg[key] += weights[i][key]
                weight_avg[key] = torch.div(weight_avg[key], len(weights))

            self.shared_adapter = weight_avg
            if self.args.algorithm == 'fdlora':
                self.shared_optimizer = torch.optim.SGD(
                    self.shared_adapter.values(),
                    lr=1.0, momentum=0.5, nesterov=True
                )
            for client in clients:
                client.shared_adapter = deepcopy(weight_avg)
                if self.args.do_profile:
                    client.download_volume.append(get_total_size(client.shared_adapter))


class Client:
    def __init__(self, idx, args, train_loader, aux_loader, eval_loader, test_loader, tokenizer):
        self.idx = idx
        self.args = args
        self.device = torch.device('cuda:{0}'.format(args.gpu))
        self.tokenizer = tokenizer

        self.staged_lam = []  # save lambda in each round (for FedAPEN)

        self.shared_model = None
        self.shared_adapter = None
        if args.algorithm in ['learned_adaptive_training', 'mutual']:  # these methods have a private model
            self.private_model = None
            self.private_adapter = None

        self.lr = ExponentialLR(args.lr, args.lr_decay)
        self.ce_loss = CausalLMLoss() if args.task_type == "causal_lm" else ClsLMLoss()
        self.kl_loss = nn.KLDivLoss(reduction='mean')
        if args.algorithm.lower() in ['learned_adaptive_training', 'mutual']:
            self.lam = torch.tensor(
                [0.5], requires_grad=True, device=self.device  # initial value is 0.5
            )
            self.optimizer_lam = torch.optim.Adam([self.lam], lr=1e-3, eps=1e-6 if args.precision == 'fp16' else 1e-8)

        self.train_loader = train_loader
        self.train_iter = iter(train_loader)
        self.aux_loader = aux_loader
        self.aux_iter = iter(aux_loader) if aux_loader is not None else None
        self.eval_loader = eval_loader
        self.test_loader = test_loader

        if self.args.do_profile:
            self.upload_volume = []
            self.download_volume = []

    def load_models(self, enable=True):
        if enable:
            set_peft_model_state_dict(self.shared_model, self.shared_adapter, adapter_name='adapter')
            if self.args.algorithm in ['learned_adaptive_training', 'mutual']:
                set_peft_model_state_dict(self.private_model, self.private_adapter, adapter_name='adapter')

    def release_models(self, enable=True):
        if enable:
            self.shared_adapter = {
                k: v.detach().cpu() for k, v in get_peft_model_state_dict(
                    self.shared_model, adapter_name='adapter', save_embedding_layers=False
                ).items()
            }
            if self.args.algorithm in ['learned_adaptive_training', 'mutual']:
                self.private_adapter = {
                    k: v.detach().cpu() for k, v in get_peft_model_state_dict(
                        self.private_model, adapter_name='adapter', save_embedding_layers=False
                    ).items()
                }

    def train_lam(self):
        """
        Learning for Adaptability (for FedAPEN)
        @return:
        """
        self.private_model = self.private_model.to(self.device)
        self.private_model.eval()
        self.shared_model = self.shared_model.to(self.device)
        self.shared_model.eval()

        aux_loader = self.aux_loader
        ce_loss = CausalLMLoss()
        for _ in range(10):
            for batch in aux_loader:
                self.optimizer_lam.zero_grad()

                data = {
                    k: v.to(self.device) for k, v in batch.items()
                    if k in ['input_ids', 'labels', 'attention_mask']
                }
                target = batch['labels'].to(self.device)
                output_private = self.private_model(**data).logits.detach()
                output_shared = self.shared_model(**data).logits.detach()

                ensemble_output = self.lam * output_private + (1 - self.lam) * output_shared
                loss = ce_loss(ensemble_output, target)

                loss.backward()
                self.optimizer_lam.step()
                torch.clip_(self.lam.data, 0.0, 1.0)

        self.staged_lam.append(torch.mean(self.lam).item())
        print('client {0} lam: {1}'.format(
            self.idx, torch.mean(self.lam).item())
        )
        
    def adaptive_fusion(self):
        """
        Adaptive Fusion (for FDLoRA)
        @return:
        """
        get_score_partial = partial(
            get_score,
            client=self
        )
        params = ng.p.Array(
            init=[0] * 2,
            upper=[1.5] * 2,
            lower=[-1.5] * 2,
        )
        optimizer = ng.optimizers.NGOpt(parametrization=params, budget=5)
        optimizer.register_callback("tell", ProgressBar(self.idx))

        weights = optimizer.minimize(get_score_partial, verbosity=0).value
        final_adapter = get_final_adapter(weights, [self.shared_adapter, self.private_adapter])
        set_peft_model_state_dict(self.shared_model, final_adapter, adapter_name='adapter')
