from datasets import load_dataset


def load_conll2003():
    dataset = load_dataset("lhoestq/conll2003")
    texts = []

    for split in ["train", "validation", "test"]:
        for sentence in dataset[split]:
            tokens = sentence["tokens"]
            texts.append(tokens)

    return texts
