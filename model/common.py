import torch

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    SwitchTransformersForConditionalGeneration
)
from data.preprocess import DefaultToken
from data.prompt import DATASET_TO_TEMPLATE
from peft import (
    LoraConfig,
    PromptTuningConfig,
    PromptTuningInit,
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
    TaskType,
    PeftMixedModel,
    get_peft_model
)

MODEL_NAME2ID = {
    'opt-350m': "facebook/opt-350m",
    'falcon-1b': "tiiuae/falcon-rw-1b",
    'llama3.2-1b': "meta-llama/Llama-3.2-1B",
    'llama3.2-3b': "meta-llama/Llama-3.2-3B",
    'gemma2-2b': "google/gemma-2-2b",
    'qwen2-0.5b': 'Qwen/Qwen2-0.5B',
    'qwen2-1.5b': 'Qwen/Qwen2-1.5B',
    'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B',
    'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B',
    'switch-base-32': 'google/switch-base-32'
}


def get_model(
    model_name: str, args,
    no_adapter=True,
    adapter_name="expert_0",
    peft_method="lora",
    tokenizer=None
):
    model_id = MODEL_NAME2ID[model_name]
    if model_name == "switch-base-32":
        model = SwitchTransformersForConditionalGeneration.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if args.precision == "fp16" else torch.float32
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if args.precision == "fp16" else torch.float32
        )

    if not no_adapter:
        if peft_method == 'lora':
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                target_modules=['q', 'v'] if model_name == 'switch-base-32' else None,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout
            )
            model = get_peft_model(model, peft_config, adapter_name=adapter_name, mixed=True)
        elif peft_method == 'prompt_tuning':
            soft_prompt = DATASET_TO_TEMPLATE[args.client_dataset_name].soft_prompt
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                prompt_tuning_init=PromptTuningInit.TEXT,
                prompt_tuning_init_text=soft_prompt,
                num_virtual_tokens=len(tokenizer(soft_prompt)["input_ids"]),
                tokenizer_name_or_path=model_id,
            )
            model = get_peft_model(model, peft_config, adapter_name=adapter_name)
        elif peft_method == 'ptuning':
            peft_config = PromptEncoderConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=20,
                encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
                encoder_hidden_size=1024
            )
            model = get_peft_model(model, peft_config, adapter_name=adapter_name)
        else:
            raise ValueError(f'Unsupported PEFT method {peft_method}!')

    return model


def get_tokenizer(args):
    model_id = MODEL_NAME2ID[args.model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    special_tokens = {}
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    tokenizer.num_added_tokens = num_added_tokens

    config = AutoConfig.from_pretrained(model_id)
    if hasattr(config, "max_position_embeddings"):
        max_pos_embs = config.max_position_embeddings
    else:
        max_pos_embs = args.max_length
    tokenizer.model_max_length = min(args.max_length, max_pos_embs)

    return tokenizer


def sync_model_with_tokenizer(model: PeftMixedModel, tokenizer):
    model.resize_token_embeddings(len(tokenizer))
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Initialize the newly-added token embedding to the mean of all token embeddings
    token_embs = model.get_input_embeddings().weight
    mean_token_emb = token_embs.mean(dim=0)
    for i in range(tokenizer.num_added_tokens):
        token_embs[-(i + 1), :] = mean_token_emb
