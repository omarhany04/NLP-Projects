import numpy as np
from collections import defaultdict, Counter

class NaiveBayesScratch:

    def __init__(self):
        self.class_priors = {}
        self.word_counts = {}
        self.class_word_totals = {}
        self.vocab = set()

    # Train
    def fit(self, texts, labels):

        total_docs = len(labels)
        label_counts = Counter(labels)

        # Class priors
        for label in label_counts:
            self.class_priors[label] = label_counts[label] / total_docs
            self.word_counts[label] = defaultdict(int)
            self.class_word_totals[label] = 0

        # Count words
        for tokens, label in zip(texts, labels):
            for word in tokens:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

        self.vocab_size = len(self.vocab)

    # Predict One
    def predict_one(self, tokens):

        scores = {}

        for label in self.class_priors:
            log_prob = np.log(self.class_priors[label])

            for word in tokens:
                word_freq = self.word_counts[label][word]

                # Laplace smoothing
                prob = (word_freq + 1) / (
                    self.class_word_totals[label] + self.vocab_size
                )

                log_prob += np.log(prob)

            scores[label] = log_prob

        return max(scores, key=scores.get)

    # Predict Many
    def predict(self, texts):
        return [self.predict_one(t) for t in texts]