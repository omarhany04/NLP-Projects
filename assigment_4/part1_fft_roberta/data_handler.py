from datasets import load_dataset

import config


def load_code_dataset(sample_size=config.DATASET_SAMPLE_SIZE, seed=config.SEED):
    dataset = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)

    if sample_size is not None:
        sample_size = min(int(sample_size), len(dataset))
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    return dataset


def build_training_text(example, eos_token=""):
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    output_text = (example.get("output") or example.get("text") or "").strip()

    parts = [f"Instruction:\n{instruction}"]
    if input_text:
        parts.append(f"Input:\n{input_text}")
    parts.append(f"Answer:\n{output_text}")

    return {"text": "\n\n".join(parts) + eos_token}


def tokenize_dataset(dataset, tokenizer, max_seq_length=config.MAX_SEQ_LENGTH):
    eos_token = tokenizer.eos_token or ""

    formatted = dataset.map(
        lambda example: build_training_text(example, eos_token=eos_token),
        remove_columns=dataset.column_names,
        desc="Formatting code-generation examples",
    )

    tokenized = formatted.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
        ),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing FFT dataset",
    )

    return tokenized
