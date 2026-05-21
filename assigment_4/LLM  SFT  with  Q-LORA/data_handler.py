from datasets import load_dataset
import config


SYSTEM_PROMPT = "You are an expert software engineer. Write functional, compilable Python code."


def load_and_sample_dataset(sample_size=config.DATASET_SAMPLE_SIZE):
    dataset = load_dataset(config.DATASET_NAME,split="train")
    sample_size = min(int(sample_size), len(dataset))
    sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))
    return sampled_dataset


def build_prompt(instruction, input_text=""):
    user_text = (instruction or "").strip()

    if input_text:
        user_text += f"\nInput details: {input_text.strip()}"

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_text}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_completion(output_text):
    output_text = (output_text or "").strip()
    return f"{output_text}<|im_end|>"


def prepare_prompt_completion_dataset(dataset):
    def format_batch(batch):
        prompts = []
        completions = []

        for instruction, input_text, output_text in zip(
            batch["instruction"],
            batch.get("input", [""] * len(batch["instruction"])),
            batch["output"],
        ):
            prompts.append(build_prompt(instruction, input_text))
            completions.append(build_completion(output_text))

        return {"prompt": prompts, "completion": completions}

    return dataset.map(
        format_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Formatting SFT prompt-completion examples",
    )


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        user_text = example['instruction'][i]

        if example.get('input') and example['input'][i]:
            user_text += f"\nInput details : {example['input'][i]}"

        assistant_text = example['output'][i]

        # format the prompt
        text  = f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n<|im_start|>user\n{user_text}\n<|im_end|>\n<|im_start|>assistant\n{assistant_text}\n<|im_end|>"
        output_texts.append(text)

    return output_texts
