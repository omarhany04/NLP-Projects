from collections import Counter, defaultdict
import math
import numpy as np

from models.hmm_constants import END_TAG, START_TAG, UNK_TOKENS
from models.hmm_helpers import (
    classify_unknown_word,
    is_valid_bio_transition,
    normalize_training_data,
)


class HMM_NER:
    def __init__(
        self,
        unk_threshold=2,
        k_trans=1.0,
        k_emit=1.0,
        use_bio_constraints=True,
        invalid_transition_penalty=1e-12,
    ):
        self.transition_counts = defaultdict(Counter)
        self.emission_counts = defaultdict(Counter)
        self.tag_counts = Counter()
        self.word_counts = Counter()

        self.tags = []
        self.vocab = set()

        self.start_tag = START_TAG
        self.end_tag = END_TAG

        self.unk_threshold = unk_threshold
        self.k_trans = k_trans
        self.k_emit = k_emit
        self.use_bio_constraints = use_bio_constraints
        self.invalid_transition_penalty = invalid_transition_penalty

        self.unk_tokens = set(UNK_TOKENS)

    def _map_word(self, word):
        if word in self.vocab:
            return word
        return classify_unknown_word(word)

    def train(self, sentences, tag_lists=None):
        sentences, tag_lists = normalize_training_data(sentences, tag_lists)

        self.transition_counts.clear()
        self.emission_counts.clear()
        self.tag_counts.clear()
        self.word_counts.clear()
        self.tags = []
        self.vocab = set()

        # Count words first
        for sentence in sentences:
            for word in sentence:
                self.word_counts[word] += 1

        # Keep only frequent words in vocab
        self.vocab = {
            word for word, count in self.word_counts.items()
            if count > self.unk_threshold
        }

        # Add special unknown classes to vocab
        self.vocab.update(self.unk_tokens)

        for sentence, tags in zip(sentences, tag_lists):
            prev_tag = self.start_tag

            for word, tag in zip(sentence, tags):
                mapped_word = (
                    word
                    if self.word_counts[word] > self.unk_threshold
                    else classify_unknown_word(word)
                )

                self.transition_counts[prev_tag][tag] += 1
                self.emission_counts[tag][mapped_word] += 1
                self.tag_counts[tag] += 1

                prev_tag = tag

            self.transition_counts[prev_tag][self.end_tag] += 1

        self.tags = sorted(self.tag_counts.keys())

    def get_transition_prob(self, prev_tag, current_tag):
        if self.use_bio_constraints and not is_valid_bio_transition(
            prev_tag, current_tag, self.start_tag, self.end_tag
        ):
            return self.invalid_transition_penalty

        count = self.transition_counts[prev_tag][current_tag]
        total = sum(self.transition_counts[prev_tag].values())

        # Possible next states: all tags + END
        num_next_states = len(self.tags) + 1
        return (count + self.k_trans) / (total + self.k_trans * num_next_states)

    def get_emission_prob(self, tag, word):
        mapped_word = self._map_word(word)

        count = self.emission_counts[tag][mapped_word]
        total = self.tag_counts[tag]
        vocab_size = len(self.vocab)

        return (count + self.k_emit) / (total + self.k_emit * vocab_size)

    def viterbi(self, sentence):
        if not sentence:
            return []

        n_words = len(sentence)
        n_tags = len(self.tags)

        dp = np.full((n_words, n_tags), -np.inf)
        backpointer = np.zeros((n_words, n_tags), dtype=int)

        # Initialization
        first_word = sentence[0]
        for s, tag in enumerate(self.tags):
            trans_p = self.get_transition_prob(self.start_tag, tag)
            emit_p = self.get_emission_prob(tag, first_word)
            dp[0, s] = math.log(trans_p) + math.log(emit_p)

        # Recursion
        for t in range(1, n_words):
            word = sentence[t]

            for s, tag in enumerate(self.tags):
                emit_p = self.get_emission_prob(tag, word)

                best_prev_score = -np.inf
                best_prev_state = 0

                for s_prev, prev_tag in enumerate(self.tags):
                    trans_p = self.get_transition_prob(prev_tag, tag)
                    score = dp[t - 1, s_prev] + math.log(trans_p) + math.log(emit_p)

                    if score > best_prev_score:
                        best_prev_score = score
                        best_prev_state = s_prev

                dp[t, s] = best_prev_score
                backpointer[t, s] = best_prev_state

        # Termination
        best_last_score = -np.inf
        best_last_state = 0

        for s, tag in enumerate(self.tags):
            trans_p = self.get_transition_prob(tag, self.end_tag)
            score = dp[n_words - 1, s] + math.log(trans_p)

            if score > best_last_score:
                best_last_score = score
                best_last_state = s

        # Backtrack
        best_path = [best_last_state]
        for t in range(n_words - 1, 0, -1):
            best_last_state = backpointer[t, best_last_state]
            best_path.insert(0, best_last_state)

        return [self.tags[i] for i in best_path]
