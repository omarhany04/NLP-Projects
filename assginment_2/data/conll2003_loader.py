from datasets import load_dataset


CONLL2003_LABELS = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]


def _extract_split(split):
    tokens = [sample["tokens"] for sample in split]
    tags = [sample["ner_tags"] for sample in split]
    return {
        "tokens": tokens,
        "tags": tags,
    }


def _normalize_label_names(label_names):
    """
    Replace generic Hugging Face-style labels with the standard CoNLL-2003 tags.
    """
    if not label_names:
        return label_names

    generic_pattern = all(label == f"LABEL_{i}" for i, label in enumerate(label_names))
    if generic_pattern and len(label_names) == len(CONLL2003_LABELS):
        return CONLL2003_LABELS.copy()

    return label_names


def _get_label_names(dataset):
    """
    Robustly extract NER label names from Hugging Face dataset features.
    """
    ner_feature = dataset["train"].features["ner_tags"]

    # Common case: Sequence(ClassLabel)
    if hasattr(ner_feature, "feature") and hasattr(ner_feature.feature, "names"):
        return _normalize_label_names(ner_feature.feature.names)

    # Sometimes the feature itself may already expose names
    if hasattr(ner_feature, "names"):
        return _normalize_label_names(ner_feature.names)

    # Fallback: inspect actual labels present and create generic names
    all_ids = set()
    for sample in dataset["train"]:
        for tag_id in sample["ner_tags"]:
            all_ids.add(tag_id)

    max_id = max(all_ids)
    fallback_names = [f"LABEL_{i}" for i in range(max_id + 1)]
    return _normalize_label_names(fallback_names)


def load_conll2003(include_validation=True):
    """
    Returns:
        all_tokens,
        train_sentences,
        train_tags,
        validation_sentences,
        validation_tags,
        test_sentences,
        test_tags,
        label_names,
        id2label,
        label2id
    """
    splits, label_names, id2label, label2id = load_conll2003_splits(
        include_validation=include_validation
    )
    all_tokens = get_all_sentences(splits, include_test=True)

    return (
        all_tokens,
        splits["train"]["tokens"],
        splits["train"]["tags"],
        splits.get("validation", {}).get("tokens"),
        splits.get("validation", {}).get("tags"),
        splits["test"]["tokens"],
        splits["test"]["tags"],
        label_names,
        id2label,
        label2id,
    )


def load_conll2003_splits(include_validation=True):
    dataset = load_dataset("lhoestq/conll2003")

    label_names = _get_label_names(dataset)
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}

    splits = {
        "train": _extract_split(dataset["train"]),
        "test": _extract_split(dataset["test"]),
    }

    if include_validation and "validation" in dataset:
        splits["validation"] = _extract_split(dataset["validation"])

    return splits, label_names, id2label, label2id


def get_all_sentences(splits, include_test=True):
    sentences = []
    for split_name in ("train", "validation", "test"):
        if split_name not in splits:
            continue
        if split_name == "test" and not include_test:
            continue
        sentences.extend(splits[split_name]["tokens"])
    return sentences
