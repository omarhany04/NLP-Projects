from collections import Counter, defaultdict
import math
import numpy as np

class HMM_NER:
    def __init__(self):
        self.transition_counts = defaultdict(Counter)
        self.emission_counts = defaultdict(Counter)
        self.tag_counts = Counter()
        self.tags = set()
        self.vocab = set()
        self.start_tag = "<START>"
        self.end_tag = "<END>"


      #compute transition and emission probabilities
    def train(self,sentences,tag_lists):
        for sentence , tags in zip(sentences,tag_lists): # pair each word sequence with its corresponding tag sequence
            prev_tag = self.start_tag # start of sentence
            self.tag_counts[self.start_tag] += 1 # count the start tag
            for word, tag in zip(sentence, tags):
                self.vocab.add(word) # add word to vocab
                self.tags.add(tag) # add tag to tag set
                self.transition_counts[prev_tag][tag] += 1 # count transition from prev_tag to current tag
                self.emission_counts[tag][word] += 1 # count emission of word given tag
                self.tag_counts[tag] += 1 # count the tag itself
                prev_tag = tag  # update prev_tag to current tag for next iteration
            self.transition_counts[prev_tag][self.end_tag] += 1 # count transition from last tag to end tag
        self.tags = list(self.tags) # convert set to list for indexing

    def get_transition_prob(self,prev_tag,current_tag):
        # Add-one smoothing
        count = self.transition_counts[prev_tag][current_tag]
        total = self.tag_counts[prev_tag]
        return (count + 1e-5) / (total + 1e-5 * len(self.tags))
    
    def get_emission_prob(self, tag, word):
        if word not in self.vocab:
            return 1e-5  # small probability for unseen words
        count = self.emission_counts[tag][word]
        total = self.tag_counts[tag]
        if count == 0:
            return 1e-7 # even smaller probability for unseen word-tag pairs
        return count / total
    

    def viterbi(self,sentence):
        n_words = len(sentence)
        n_tags = len(self.tags)
        # viterbi[i][j] = max prob of tag i at position j
        viterbi = np.zeros((n_words, n_tags))
        backpointer = np.zeros((n_words, n_tags), dtype=int)
        # Initialization
        word = sentence[0]

        for s in range(n_tags):
            tag = self.tags[s]  # get the tag corresponding to index s
            trans_p = self.get_transition_prob(self.start_tag, tag) # transition from start tag to current tag
            emit_p = self.get_emission_prob(tag, word) # emission of first word given current tag
            viterbi[0][s] = math.log(trans_p) + math.log(emit_p) # using log probabilities to avoid underflow
            backpointer[0][s] = 0

        # Recursion
        for t in range(1,n_words):
            word = sentence[t]

            for s in range(n_tags):
                tag = self.tags[s]
                emiss_p = self.get_emission_prob(tag, word) # emission of current word given current tag
                max_prob = -float('inf') # initialize max_prob to negative infinity for comparison
                best_prev_state = 0

                for s_prev in range(n_tags):
                    prev_tag = self.tags[s_prev]     # get the previous tag corresponding to index s_prev
                    trans_p = self.get_transition_prob(prev_tag, tag)  # transition from previous tag to current tag
                    prob = viterbi[t-1][s_prev] + math.log(trans_p) + math.log(emiss_p) # total probability of the best path to previous tag plus transition and emission probabilities

                    if prob > max_prob:
                        max_prob = prob
                        best_prev_state = s_prev
                viterbi[t][s] = max_prob
                backpointer[t][s] = best_prev_state


        # Termination
        best_path_prob = -float('inf')
        best_last_tag = 0
        for s in range(n_tags):
            tag = self.tags[s]
            trans_p = self.get_transition_prob(tag, self.end_tag) # transition from current tag to end tag
            prob = viterbi[n_words-1][s] + math.log(trans_p)  # calculating final score
            if prob > best_path_prob:
                best_path_prob = prob
                best_last_tag = s
        # Backtrace
        best_path = [best_last_tag]
        
        for t in range(n_words-1,0,-1):  # start from last word and move backwards
            best_last_tag = backpointer[t][best_last_tag] # update best_last_tag to the previous tag in the best path
            best_path.insert(0, best_last_tag) # insert the best_last_tag at the beginning of best_path to build the path in correct order
        return [self.tags[i] for i in best_path]