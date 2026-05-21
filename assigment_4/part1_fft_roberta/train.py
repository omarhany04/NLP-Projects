import argparse
import inspect
import os
from dataclasses import fields, is_dataclass

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

import config
from data_handler import load_code_dataset, tokenize_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Part I: full fine-tuning baseline with XLM-RoBERTa + causal LM head."
    )
    parser.add_argument("--model_id", default=config.BASE_MODEL_ID)
    parser.add_argument("--sample_size", type=int, default=config.DATASET_SAMPLE_SIZE)
    parser.add_argument("--max_seq_length", type=int, default=config.MAX_SEQ_LENGTH)
    parser.add_argument("--output_dir", default=config.OUTPUT_DIR)
    parser.add_argument("--final_model_dir", default=config.FINAL_MODEL_DIR)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--strict_bf16",
        action="store_true",
        help=(
            "Force TrainingArguments(bf16=True). By default the script keeps the "
            "assignment BF16 setting but falls back to fp16 on GPUs such as T4 that "
            "do not expose CUDA BF16 support."
        ),
    )
    return parser.parse_args()


def precision_flags(strict_bf16=False):
    if not config.BF16:
        return {"bf16": False, "fp16": False}

    if strict_bf16:
        return {"bf16": True, "fp16": False}

    cuda_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return {"bf16": cuda_bf16, "fp16": torch.cuda.is_available() and not cuda_bf16}


def filter_training_args(kwargs):
    if is_dataclass(TrainingArguments):
        allowed = {field.name for field in fields(TrainingArguments)}
    else:
        allowed = set(inspect.signature(TrainingArguments.__init__).parameters)

    return {key: value for key, value in kwargs.items() if key in allowed}


def build_trainer(model, training_args, train_dataset, data_collator, tokenizer):
    trainer_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    return Trainer(**trainer_kwargs)


def load_roberta_causal_lm(model_id):
    model_config = AutoConfig.from_pretrained(model_id)

    # XLM-R/RoBERTa is encoder-style by default; this adapts it for causal LM fine-tuning.
    model_config.is_decoder = True
    model_config.add_cross_attention = False

    model = AutoModelForCausalLM.from_pretrained(model_id, config=model_config)
    model.config.use_cache = False

    return model


def main():
    args = parse_args()
    set_seed(config.SEED)

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", config.PROJECT_NAME)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    tokenizer.model_max_length = args.max_seq_length

    dataset = load_code_dataset(sample_size=args.sample_size, seed=config.SEED)
    train_dataset = tokenize_dataset(dataset, tokenizer, max_seq_length=args.max_seq_length)

    model = load_roberta_causal_lm(args.model_id)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        **filter_training_args(
            {
                "output_dir": args.output_dir,
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
                "report_to": [] if args.no_wandb else [config.REPORT_TO],
                "run_name": config.RUN_NAME,
                "remove_unused_columns": False,
                **precision_flags(strict_bf16=args.strict_bf16),
            }
        )
    )

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.final_model_dir)
    tokenizer.save_pretrained(args.final_model_dir)


if __name__ == "__main__":
    main()
