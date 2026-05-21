import argparse
import inspect
import os
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from trl import DPOTrainer

try:
    from trl import DPOConfig
except ImportError:
    DPOConfig = None

import config
from data_handler import load_truthy_dpo_dataset, prepare_dpo_dataset, resolve_sft_adapter_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Part III: DPO alignment beta sweep from the Part II Qwen SFT adapter."
    )
    parser.add_argument("--sft_adapter_path", default=None)
    parser.add_argument("--output_root", default=config.OUTPUT_ROOT)
    parser.add_argument("--sample_size", type=int, default=config.DPO_SAMPLE_SIZE)
    parser.add_argument("--betas", nargs="+", type=float, default=config.BETAS)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def precision_flags():
    cuda_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return {"bf16": cuda_bf16, "fp16": torch.cuda.is_available() and not cuda_bf16}


def allowed_kwargs(cls):
    if cls is None:
        return set()
    if is_dataclass(cls):
        return {field.name for field in fields(cls)}
    return set(inspect.signature(cls.__init__).parameters)


def filter_kwargs(cls, kwargs):
    allowed = allowed_kwargs(cls)
    return {key: value for key, value in kwargs.items() if key in allowed}


def load_tokenizer(adapter_path):
    tokenizer_path = adapter_path if Path(adapter_path, "tokenizer_config.json").exists() else config.BASE_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_sft_peft_model(adapter_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=compute_dtype(),
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype(),
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=True,
        adapter_name=config.TRAIN_ADAPTER_NAME,
    )
    model.load_adapter(
        adapter_path,
        adapter_name=config.REFERENCE_ADAPTER_NAME,
        is_trainable=False,
    )
    model.set_adapter(config.TRAIN_ADAPTER_NAME)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model


def build_training_args(output_dir, beta, no_wandb):
    args_cls = DPOConfig or TrainingArguments
    base_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": config.PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
        "num_train_epochs": config.NUM_TRAIN_EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "optim": config.OPTIMIZER,
        "gradient_checkpointing": config.GRADIENT_CHECKPOINTING,
        "lr_scheduler_type": config.LR_SCHEDULER_TYPE,
        "warmup_ratio": config.WARMUP_RATIO,
        "logging_steps": config.LOGGING_STEPS,
        "save_strategy": config.SAVE_STRATEGY,
        "report_to": [] if no_wandb else [config.REPORT_TO],
        "run_name": f"{config.RUN_NAME_PREFIX}_beta_{beta}",
        "remove_unused_columns": False,
        "max_prompt_length": config.MAX_PROMPT_LENGTH,
        "max_length": config.MAX_LENGTH,
        "beta": beta,
        "model_adapter_name": config.TRAIN_ADAPTER_NAME,
        "ref_adapter_name": config.REFERENCE_ADAPTER_NAME,
        **precision_flags(),
    }
    return args_cls(**filter_kwargs(args_cls, base_kwargs))


def build_trainer(model, tokenizer, train_dataset, training_args, beta):
    trainer_params = inspect.signature(DPOTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": train_dataset,
    }

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    # Compatibility with older TRL versions where these live on DPOTrainer.
    compatibility_values = {
        "beta": beta,
        "max_prompt_length": config.MAX_PROMPT_LENGTH,
        "max_length": config.MAX_LENGTH,
        "model_adapter_name": config.TRAIN_ADAPTER_NAME,
        "ref_adapter_name": config.REFERENCE_ADAPTER_NAME,
    }
    for key, value in compatibility_values.items():
        if key in trainer_params and key not in allowed_kwargs(type(training_args)):
            trainer_kwargs[key] = value

    return DPOTrainer(**trainer_kwargs)


def train_one_beta(beta, raw_dataset, adapter_path, output_root, no_wandb):
    beta_label = str(beta).replace(".", "_")
    output_dir = Path(output_root) / f"beta_{beta_label}"
    final_dir = output_dir / "final_adapter"

    tokenizer = load_tokenizer(adapter_path)
    train_dataset = prepare_dpo_dataset(raw_dataset, tokenizer)
    model = load_sft_peft_model(adapter_path)
    training_args = build_training_args(output_dir=output_dir, beta=beta, no_wandb=no_wandb)
    trainer = build_trainer(model, tokenizer, train_dataset, training_args, beta=beta)

    trainer.train()
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    set_seed(config.SEED)

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", config.PROJECT_NAME)

    adapter_path = resolve_sft_adapter_path(args.sft_adapter_path)
    raw_dataset = load_truthy_dpo_dataset(sample_size=args.sample_size, seed=config.SEED)

    for beta in args.betas:
        train_one_beta(
            beta=beta,
            raw_dataset=raw_dataset,
            adapter_path=adapter_path,
            output_root=args.output_root,
            no_wandb=args.no_wandb,
        )


if __name__ == "__main__":
    main()
