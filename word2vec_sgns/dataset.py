import random


class SkipGramDataset:

    def __init__(self, sentences, window_size):

        self.data = []
        self.window = window_size

        for sent in sentences:

            for i, target in enumerate(sent):

                start = max(0, i - window_size)
                end = min(len(sent), i + window_size + 1)

                for j in range(start, end):

                    if i != j:
                        context = sent[j]
                        self.data.append((target, context))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
