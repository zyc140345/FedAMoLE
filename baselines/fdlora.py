import torch
import nevergrad.common.typing as tp
from nevergrad.optimization import base
from peft import set_peft_model_state_dict


def get_regular(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares


def get_loss(client):
    """
    Get the loss of the model on the example dataset. Usually the example dataset only contains a few examples.
    """
    total_loss = 0
    num_examples = 0
    with torch.inference_mode():
        for _ in range(5):
            try:
                batch = next(client.aux_iter)
            except StopIteration:
                client.aux_iter = iter(client.aux_loader)
                batch = next(client.aux_iter)

            batch = {
                k: v.to(client.device) for k, v in batch.items()
                if k in ["input_ids", "labels", "attention_mask"]
            }
            outputs = client.shared_model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            num_examples += len(batch["input_ids"])

    # average loss over the number of examples
    return total_loss / num_examples


def get_score(weights, client):
    # reload the model with the composed adapter
    final_adapter = get_final_adapter(weights, [client.shared_adapter, client.private_adapter])
    set_peft_model_state_dict(client.shared_model, final_adapter, adapter_name="adapter")

    # minimize the metric
    loss = get_loss(client)
    # L1 regularization term
    metric_val = loss + get_regular(weights)

    return metric_val


def get_final_adapter(weights, adapters):
    final_adapter = {}
    for i, adapter in enumerate(adapters):
        for key in adapter.keys():
            if key not in final_adapter:
                final_adapter[key] = weights[i] * adapter[key]
            else:
                final_adapter[key] += weights[i] * adapter[key]
    return final_adapter


class ProgressBar:
    """Progress bar to register as callback in an optimizer"""

    def __init__(self, client_id: int) -> None:
        self._progress_bar: tp.Any = None
        self._current = 0
        self._client_id = client_id

    def __call__(self, optimizer: base.Optimizer, *args: tp.Any, **kwargs: tp.Any) -> None:
        if self._progress_bar is None:
            # pylint: disable=import-outside-toplevel
            try:
                from tqdm import tqdm  # Inline import to avoid additional dependency
            except ImportError as e:
                raise ImportError(
                    f"{self.__class__.__name__} requires tqdm which is not installed by default "
                    "(pip install tqdm)"
                ) from e
            self._progress_bar = tqdm(desc=f'Node{self._client_id} Adaptive Fusion')
            self._progress_bar.total = optimizer.budget
            self._progress_bar.update(self._current)
        self._progress_bar.update(1)
        self._current += 1

    def __getstate__(self) -> tp.Dict[str, tp.Any]:
        """Used for pickling (tqdm is not picklable)"""
        state = dict(self.__dict__)
        state["_progress_bar"] = None
        return state
