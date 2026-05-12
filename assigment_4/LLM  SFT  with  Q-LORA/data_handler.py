from datasets import load_dataset
import config

def load_and_sample_dataset():
    dataset = load_dataset(config.DATASET_NAME,split="train")
    sampled_dataset = dataset.shuffle(seed=42).select(range(config.DATASET_SAMPLE_SIZE))
    return sampled_dataset

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        system_prompt = "You are an expert software engineer. Write functional, compilable Python code."
        user_text = example['instruction'][i]

        if example.get('input') and example['input'][i]:
            user_text += f"\nInput details : {example['input'][i]}"

        assistant_text = example['output'][i]

        # format the prompt
        text  = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n<|im_start|>user\n{user_text}\n<|im_end|>\n<|im_start|>assistant\n{assistant_text}\n<|im_end|>"
        output_texts.append(text)

    return output_texts