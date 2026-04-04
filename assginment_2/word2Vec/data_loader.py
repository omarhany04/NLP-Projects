from datasets import load_dataset


def load_conll2003():
    dataset = load_dataset("lhoestq/conll2003")
    train_texts = []
    for sentence in dataset["train"]:
        tokens = sentence["tokens"]
        train_texts.append(tokens)
    test_data = dataset['test']
    train_sentences = dataset['train']
    train_tags = train_sentences['ner_tags']
    test_sentences = test_data['tokens']
    test_true_tags = test_data['ner_tags']
    return train_texts,train_sentences, train_tags, test_sentences, test_true_tags