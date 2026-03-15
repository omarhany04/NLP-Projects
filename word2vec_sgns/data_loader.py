from datasets import load_dataset


def load_conll2003():
    dataset = load_dataset("lhoestq/conll2003")

    sentences = []

    for split in ["train", "validation", "test"]:
        for example in dataset[split]:
            tokens = example["tokens"]
            sentences.append(tokens)

    return sentences
