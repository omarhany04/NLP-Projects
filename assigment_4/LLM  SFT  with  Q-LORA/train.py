import torch
import wandb


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)


from peft import LoraConfig
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM

import config
from data_handler import load_and_sample_dataset,formatting_prompts_func

def main():

    wandb.init(project="LLM_Alignment_Assignment_4", name="Part2_SFT_Qwen")
    dataset = load_and_sample_dataset()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID,trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = config.MAX_SEQ_LENGTH

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
    bnb_4bit_compute_dtype = torch.bfloat16
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
    
    response_template = "<|im_start|>assistant\n"

    data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=response_template)

    training_args = TrainingArguments(
        output_dir="./results_sft_qwen",
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        optim=config.OPTIMIZER,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
        args=training_args,
    )

    trainer.train()

    trainer.model.save_pretrained("./sft_qwen_model")
    tokenizer.save_pretrained("./sft_qwen_model")

    wandb.finish()

if __name__ == "__main__":
    main()
