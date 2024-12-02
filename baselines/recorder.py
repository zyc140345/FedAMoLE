import torch
import os
import json
import numpy as np


class Recorder:
    def __init__(self, args, evaluator):
        self.args = args
        self.evaluator = evaluator
        self.train_record = [[] for _ in range(args.client_num)]

        if args.task_type == 'cls_lm':
            self.acc_best = torch.zeros(self.args.client_num)
            self.avg_acc_best = 0
        else:
            self.loss_best = torch.zeros(self.args.client_num)
            self.avg_loss_best = 0
        self.get_a_better = [False for _ in range(self.args.client_num)]

        self.server = None
        self.clients = None

    def test(self, client):
        if self.args.algorithm in ['learned_adaptive_training', 'mutual']:
            self.evaluator.test_ensemble_model(client)
        else:
            self.evaluator.test_single_model(client)

    def eval(self, client):
        if self.args.algorithm in ['learned_adaptive_training', 'mutual']:
            self.evaluator.eval_ensemble_model(client)
        else:
            self.evaluator.eval_single_model(client)

    def add_train_record(self, record: dict):
        idx = record['idx']
        self.train_record[idx].append(record)

    def register_server(self, server):
        self.server = server

    def register_clients(self, clients):
        self.clients = clients

    def summary(self, cur_round):
        if self.args.algorithm in ['learned_adaptive_training', 'mutual']:
            lam = [client.staged_lam for client in self.clients]
            if self.args.task_type == 'cls_lm':
                print(f'The {cur_round + 1:d}-th round, ', end='')
                if self.args.log_level == 'detailed':
                    print(
                        'Average Private Loss: {:.3f}, '.format(np.mean([
                            self.evaluator.test_loss_mutual_private[idx][-1]
                            for idx in range(self.args.client_num)
                        ])) +
                        'Average Private ACC: {:.3f}%, '.format(np.mean([
                            self.evaluator.test_acc_mutual_private[idx][-1]
                            for idx in range(self.args.client_num)
                        ])) +
                        'Average Shared Loss: {:.3f}, '.format(np.mean([
                            self.evaluator.test_loss_mutual_shared[idx][-1]
                            for idx in range(self.args.client_num)
                        ])) +
                        'Average shared ACC: {:.3f}%, '.format(np.mean([
                            self.evaluator.test_acc_mutual_shared[idx][-1]
                            for idx in range(self.args.client_num)
                        ])),
                        end=''
                    )
                if self.args.algorithm == 'mutual' or self.args.log_level == 'detailed':
                    print(
                        'Average Ensemble Loss: {:.3f}, '.format(np.mean([
                            self.evaluator.test_loss_mutual_ensemble[idx][-1]
                            for idx in range(self.args.client_num)
                        ])) +
                        'Average Ensemble ACC: {:.3f}%'.format(np.mean([
                            self.evaluator.test_acc_mutual_ensemble[idx][-1]
                            for idx in range(self.args.client_num)
                        ])),
                        end='\n' if self.args.algorithm == 'mutual' else ', '
                    )
                if self.args.algorithm == 'learned_adaptive_training':
                    print(
                        'Average Adaptive Ensemble Loss: {:.3f}, '.format(np.mean([
                            self.evaluator.test_loss_mutual_ensemble_adaptive[idx][-1]
                            for idx in range(self.args.client_num)
                        ])) +
                        'Average Adaptive Ensemble ACC: {:.3f}%'.format(np.mean([
                            self.evaluator.test_acc_mutual_ensemble_adaptive[idx][-1]
                            for idx in range(self.args.client_num)
                        ]))
                    )
            else:
                print(f'The {cur_round + 1:d}-th round, ', end='')
                if self.args.log_level == 'detailed':
                    print(
                        'Average Private Loss: {:.3f}, '.format(np.mean([
                            self.evaluator.test_loss_mutual_private[idx][-1]
                            for idx in range(self.args.client_num)
                        ])) +
                        'Average Shared Loss: {:.3f}, '.format(np.mean([
                            self.evaluator.test_loss_mutual_shared[idx][-1]
                            for idx in range(self.args.client_num)
                        ])),
                        end=''
                    )
                if self.args.algorithm == 'mutual' or self.args.log_level == 'detailed':
                    print(
                        'Average Ensemble Loss: {:.3f}'.format(np.mean([
                            self.evaluator.test_loss_mutual_ensemble[idx][-1]
                            for idx in range(self.args.client_num)
                        ])),
                        end='\n' if self.args.algorithm == 'mutual' else ', '
                    )
                if self.args.algorithm == 'learned_adaptive_training':
                    print(
                        'Average Adaptive Ensemble Loss: {:.3f}'.format(np.mean([
                            self.evaluator.test_loss_mutual_ensemble_adaptive[idx][-1]
                            for idx in range(self.args.client_num)
                        ]))
                    )

            with open(os.path.join(self.args.log_root, 'summary.json'), 'w') as f:
                dic = {k: v for k, v in self.evaluator.__dict__.items() if k != 'args'}
                dic.update({"train_record": self.train_record, "lam": lam})
                json.dump(dic, f, indent=4)
        else:
            if self.args.task_type == 'cls_lm':
                print('The {:d}-th round, Average Loss: {:.3f}, Average ACC: {:.3f}%!'.format(
                    cur_round + 1,
                    np.mean([self.evaluator.test_loss[idx][-1] for idx in range(self.args.client_num)]),
                    np.mean([self.evaluator.test_acc[idx][-1] for idx in range(self.args.client_num)])
                ))
            else:
                print('The {:d}-th round, Average Loss: {:.3f}!'.format(
                    cur_round + 1,
                    np.mean([self.evaluator.test_loss[idx][-1] for idx in range(self.args.client_num)])
                ))

            with open(os.path.join(self.args.log_root, 'summary.json'), 'w') as f:
                dic = {k: v for k, v in self.evaluator.__dict__.items() if k != 'args'}
                dic.update({"train_record": self.train_record})
                json.dump(dic, f, indent=4)
