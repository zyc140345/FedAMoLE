import torch
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass


@dataclass
class EnsembleCausalLMOutputWithPast(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]], Tuple[Tuple[torch.FloatTensor]]]] = None
    hidden_states: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]], Tuple[Tuple[torch.FloatTensor]]]] = None
    attentions: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]], Tuple[Tuple[torch.FloatTensor]]]] = None


class EnsembleCausalLMModel(PreTrainedModel, GenerationMixin):
    _supports_sdpa = True

    def __init__(self, private_model, shared_model, lam):
        super().__init__(private_model.config)
        self.generation_config.pad_token_id = private_model.generation_config.pad_token_id

        self.private_model = private_model
        self.shared_model = shared_model
        self.lam = lam

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]], Tuple[Tuple[torch.FloatTensor]]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # model outputs consists of (logits, layer_state, hidden, attn)
        private_outputs = self.private_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # head_mask=head_mask,
            past_key_values=past_key_values[0] if past_key_values is not None else None,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        shared_outputs = self.shared_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # head_mask=head_mask,
            past_key_values=past_key_values[1] if past_key_values is not None else None,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not isinstance(private_outputs, tuple):
            private_outputs = (
                private_outputs.logits, private_outputs.past_key_values,
                private_outputs.hidden_states, private_outputs.attentions
            )
        if not isinstance(shared_outputs, tuple):
            shared_outputs = (
                shared_outputs.logits, shared_outputs.past_key_values,
                shared_outputs.hidden_states, shared_outputs.attentions
            )
        ensemble_logits = self.lam * private_outputs[0] + (1 - self.lam) * shared_outputs[0]
        ensemble_outputs = (
            ensemble_logits,
            (private_outputs[1], shared_outputs[1]),
            (private_outputs[2], shared_outputs[2]),
            (private_outputs[3], shared_outputs[3])
        )

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(ensemble_logits.device)
            # Shift so that tokens < n predict n
            shift_ensemble_logits = ensemble_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_ensemble_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (ensemble_logits,) + ensemble_outputs[1:]
            return (loss,) + output if loss is not None else output

        return EnsembleCausalLMOutputWithPast(
            loss=loss,
            logits=ensemble_outputs[0],
            past_key_values=ensemble_outputs[1],
            hidden_states=ensemble_outputs[2],
            attentions=ensemble_outputs[3]
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        The `generate` method of the `GenerationMixin` class calls the `_validate_model_class` method to
        check if the model can generate. This method further invokes the `can_generate` method,
        which is an abstract method that needs to be implemented by subclasses.

        For the `PreTrainedModel` class, a subclass of `GenerationMixin`,
        the criteria in the `can_generate` method are that the model must implement both
        the `prepare_inputs_for_generation` and `generate` methods.

        Therefore, to ensure that a custom subclass of `PreTrainedModel` can use the `generate` method of
        the parent `GenerationMixin` class, it must implement the `prepare_inputs_for_generation` method.
        """

        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        The `generate` method of the `GenerationMixin` class calls the `_reorder_cache` method to
        cache the attention key-values (`past_key_values`) when the model generates each token.

        This is an abstract method that requires implementation by subclasses.
        """

        private_reordered_past = ()
        shared_reordered_past = ()
        for layer_past in past_key_values[0]:
            private_reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        for layer_past in past_key_values[1]:
            shared_reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return private_reordered_past, shared_reordered_past
