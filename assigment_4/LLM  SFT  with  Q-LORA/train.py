import argparse
import inspect
import os
from dataclasses import fields, is_dataclass

import torch
import wandb


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)


from peft import LoraConfig
from trl import SFTTrainer

try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None

import config
from data_handler import load_and_sample_dataset, prepare_prompt_completion_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Part II: Qwen SFT with Q-LoRA.")
    parser.add_argument("--sample_size", type=int, default=config.DATASET_SAMPLE_SIZE)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def allowed_kwargs(cls):
    if cls is None:
        return set()
    if is_dataclass(cls):
        return {field.name for field in fields(cls)}
    return set(inspect.signature(cls.__init__).parameters)


def filter_kwargs(cls, kwargs):
    allowed = allowed_kwargs(cls)
    return {key: value for key, value in kwargs.items() if key in allowed}


def build_training_args(no_wandb=False):
    args_cls = SFTConfig or TrainingArguments
    args_kwargs = {
        "output_dir": "./results_sft_qwen",
        "overwrite_output_dir": True,
        "per_device_train_batch_size": config.PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
        "num_train_epochs": config.NUM_TRAIN_EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "optim": config.OPTIMIZER,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": [] if no_wandb else ["wandb"],
        "run_name": "Part2_SFT_Qwen",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "max_length": config.MAX_SEQ_LENGTH,
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "completion_only_loss": True,
        "packing": False,
    }
    return args_cls(**filter_kwargs(args_cls, args_kwargs))


def build_sft_trainer(model, tokenizer, dataset, peft_config, training_args):
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": peft_config,
        "args": training_args,
    }

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    compatibility_values = {
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "packing": False,
    }
    for key, value in compatibility_values.items():
        if key in trainer_params and key not in allowed_kwargs(type(training_args)):
            trainer_kwargs[key] = value

    return SFTTrainer(**trainer_kwargs)

def main():
    args = parse_args()

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        wandb.init(project="LLM_Alignment_Assignment_4", name="Part2_SFT_Qwen")

    dataset = prepare_prompt_completion_dataset(load_and_sample_dataset(sample_size=args.sample_size))
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID,trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = config.MAX_SEQ_LENGTH

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype(),
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
    )

    peft_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=32,
        target_modules=config.TARGET_MODULES,
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    training_args = build_training_args(no_wandb=args.no_wandb)

    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        peft_config=peft_config,
        training_args=training_args,
    )

    trainer.train()

    trainer.model.save_pretrained("./sft_qwen_model")
    tokenizer.save_pretrained("./sft_qwen_model")

    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
