from pathlib import Path

from datasets import concatenate_datasets, load_dataset

import config


DEFAULT_SYSTEM_PROMPT = "You are a careful, truthful, and safe assistant."


def resolve_sft_adapter_path(explicit_path=None):
    candidates = [explicit_path] if explicit_path else config.SFT_ADAPTER_CANDIDATES

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)

    searched = "\n".join(f"- {candidate}" for candidate in candidates if candidate)
    raise FileNotFoundError(
        "Could not find the Part II SFT adapter. Train Part II first or pass "
        f"--sft_adapter_path explicitly. Searched:\n{searched}"
    )


def oversample_to_size(dataset, target_size, seed=config.SEED):
    if len(dataset) == 0:
        raise ValueError("Cannot oversample an empty dataset.")

    pieces = []
    full_repeats = target_size // len(dataset)
    remainder = target_size % len(dataset)

    for repeat_idx in range(full_repeats):
        pieces.append(dataset.shuffle(seed=seed + repeat_idx))

    if remainder:
        pieces.append(dataset.shuffle(seed=seed + full_repeats).select(range(remainder)))

    return concatenate_datasets(pieces)


def load_truthy_dpo_dataset(
    sample_size=config.DPO_SAMPLE_SIZE,
    seed=config.SEED,
    oversample_if_needed=config.OVERSAMPLE_IF_NEEDED,
):
    dataset = load_dataset(config.DPO_DATASET_NAME, split=config.DPO_DATASET_SPLIT)
    sample_size = int(sample_size)

    if sample_size <= len(dataset):
        return dataset.shuffle(seed=seed).select(range(sample_size))

    if not oversample_if_needed:
        raise ValueError(
            f"Requested {sample_size} DPO rows, but {config.DPO_DATASET_NAME} "
            f"only has {len(dataset)} rows."
        )

    return oversample_to_size(dataset, target_size=sample_size, seed=seed)


def _chat_prompt(tokenizer, system_text, user_text):
    messages = [
        {"role": "system", "content": system_text.strip() or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_text.strip()},
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        f"<|im_start|>system\n{messages[0]['content']}\n<|im_end|>\n"
        f"<|im_start|>user\n{messages[1]['content']}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _completion(text, eos_token):
    text = (text or "").strip()
    if eos_token and not text.endswith(eos_token):
        return text + eos_token
    return text


def prepare_dpo_dataset(dataset, tokenizer):
    eos_token = tokenizer.eos_token or "<|im_end|>"

    def format_batch(batch):
        batch_size = len(next(iter(batch.values())))
        prompts = []
        chosen = []
        rejected = []

        systems = batch.get("system", [DEFAULT_SYSTEM_PROMPT] * batch_size)
        prompt_values = batch.get("prompt") or batch.get("question") or batch.get("input")
        if prompt_values is None:
            raise KeyError("DPO dataset must contain a prompt, question, or input column.")

        for idx in range(batch_size):
            prompts.append(_chat_prompt(tokenizer, systems[idx] or DEFAULT_SYSTEM_PROMPT, prompt_values[idx]))
            chosen.append(_completion(batch["chosen"][idx], eos_token))
            rejected.append(_completion(batch["rejected"][idx], eos_token))

        return {"prompt": prompts, "chosen": chosen, "rejected": rejected}

    return dataset.map(
        format_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Formatting DPO preference pairs",
    )
