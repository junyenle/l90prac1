# CONSTANTS AND CONFIGURATION
# data locations and splitting
root = '/usr/groups/mphil/L90/data-tagged'
data_positive = root + '/POS'
data_negative = root + '/NEG'
test_data_start = 900

# constants
frequency_cutoff_unigram = 0
frequency_cutoff_bigram = 0
k = 1

# internal imports
from utils import get_words, open_file

# external imports
import numpy as np
from collections import Counter

# class definition
class NaiveBayesProductRater:
    def __init__(self):
        X_train_pos, X_test_pos = get_data_paths(data_positive)  
        X_train_neg, X_test_neg = get_data_paths(data_negative)
        vocab = set()
        for path in X_train_pos + X_train_neg:
            message = utils.open_file(path)
            words = utils.get_words(message)
            vocab = vocab.union(set(words))
        self.vocab_size = len(vocab)
        self.p_word_given_pos = dict()
        self.p_word_given_neg = dict()
        self.p_pos = 0
        self.p_neg = 0
        self.k = k
        self.grams = 1 # set to 2 for bigrams
    
    def p_word_given_class(self, data_files, vocab_size):
        """
        helper function
        return dictionary representation of P(word | class)
        """
        #TODO: bigram trainig
        word_counter = Counter()
        for file in data_files:
            for word in open_file(file).split():
                word_counter[word] += 1
        word_counter["UNK"] = 0
        total_count = sum(word_counter.values())
        for word in word_counter:
            word_counter[word] = (word_counter[word] + self.k)/ (total_count + vocab_size + self.k)
        p_word_given_class = dict(word_counter)
        return p_word_given_class

    def p_class_given_input(self, input_path, p_word_given_class, p_class):
        """
        helper function
        return P(class | input)
        """
        #TODO: bigram predicting
        words = open_file(input_path).split()
        p_class_given_input = 0
        for word in set(words):
            if word not in p_word_given_class:
                word = "UNK"
            p_class_given_input += np.log(p_word_given_class[word])
        p_class_given_input += np.log(p_class)
        return p_class_given_input
        
    def train(self, X_train_pos, X_train_neg):
        # compute P(word | positive) and P(word | negative)
        self.p_word_given_pos = self.p_word_given_class(X_train_pos, self.vocab_size)
        self.p_word_given_neg = self.p_word_given_class(X_train_neg, self.vocab_size)
        # compute P(positive) and P(negative_
        self.p_pos = (len(X_train_pos + self.k) / (len(X_train_pos) + len(X_train_neg) + 2 * self.k)        
        self.p_neg = (len(X_train_neg + self.k) / (len(X_train_neg) + len(X_train_pos) + 2 * self.k)    

    def predict(self, X_sample):
        p_pos = self.p_class_given_input(X_instance_path, self.p_word_given_pos, self.p_pos)        
        p_neg = self.p_class_given_input(X_instance_path, self.p_word_given_neg, self.p_neg)
        if p_pos > p_neg:
            return 'pos'
        else:
            return 'neg'
        
    def evaluate(self, X, ground_truth):
        count_correct = 0
        for sample in X:
            if self.predict(X) == ground_truth):
                count_correct += 1
        return float(count_correct)/len(X)