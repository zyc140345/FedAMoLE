from torch import nn
from torch.nn import CrossEntropyLoss
from model.common import DefaultToken


class CausalLMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # logits: (batch_size, seq_len, vocab_size)
        # labels: (batch_size, seq_len)

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        return loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))


class ClsLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels, classes):
        # logits: (batch_size * num_choice, seq_len, vocab_size)
        # labels: (batch_size, num_choice, seq_len)

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != DefaultToken.IGNORE_INDEX.value
        # Flatten the tokens
        loss_all = self.loss_fct(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1)
        ).view_as(shift_labels)
        classes = classes.view(classes.shape[0], 1, 1).expand(-1, -1, loss_all.shape[-1])
        loss = loss_all.gather(1, classes).sum() / mask.gather(1, classes).sum()
        loss_all = loss_all.sum(dim=-1) / mask.sum(dim=-1)
        return loss, loss_all


def get_loss_fn(task_type):
    if task_type == 'causal_lm':
        loss_fn = None
    elif task_type == 'cls_lm':
        loss_fn = ClsLMLoss()
    else:
        raise ValueError(f"Task Type {task_type} not supported")
    return loss_fn
