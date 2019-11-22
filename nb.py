
# CONSTANTS AND CONFIGURATION
# data locations and splitting
root = '/mnt/c/Users/Jun/Downloads/data-tagged-20191111T010525Z-001/data-tagged'
# root = '/usr/groups/mphil/L90/data-tagged'

data_positive = root + '/POS'
data_negative = root + '/NEG'
test_data_start = 900

# constants
unigrams = True
bigrams = True
frequency_cutoff_unigram = 4
frequency_cutoff_bigram = 7
k = 0
numfolds = 10

# internal imports
from utils import get_words, open_file, get_data_paths

# external imports
import math
from collections import Counter

# class definition
class NaiveBayesProductRater:
    def __init__(self, fold):
        self.fold = fold
        #print('fold {}'.format(fold))
        #print('getting data paths')
        self.X_train_pos, self.X_test_pos = get_data_paths(data_positive, test_data_start, numfolds, self.fold)  
        self.X_train_neg, self.X_test_neg = get_data_paths(data_negative, test_data_start, numfolds, self.fold)
        #print('building vocabulary')
        self.vocab = Counter()
        # filtering words under threshold
        words_to_delete = set()
        ft = 0
        if bigrams:
        #    print('generating vocabulary for bigrams')
            for path in self.X_train_pos + self.X_train_neg:
                message = open_file(path)
                words = get_words(message)
                for i, word in enumerate(words):
                    if i != 0:
                        bigram = words[i-1] + ' ' + word
                        self.vocab[bigram] += 1
            ft = frequency_cutoff_bigram
            for word in self.vocab:
                if self.vocab[word] < ft:
                    words_to_delete.add(word)
        if unigrams:     
        #    print('generating vocabulary for unigrams')
            for path in self.X_train_pos + self.X_train_neg:
                message = open_file(path)
                words = get_words(message)
                for word in words:
                    self.vocab[word] += 1
            ft = frequency_cutoff_unigram
            for word in self.vocab:
                if self.vocab[word] < ft:
                    words_to_delete.add(word)
        for word in words_to_delete:
            del(self.vocab[word])
        #print('initializing rest of variables')
        self.vocab_size = len(set(self.vocab))
        self.p_word_given_pos = dict()
        self.p_word_given_neg = dict()
        self.p_pos = 0
        self.p_neg = 0
        self.k = k
        #print('initialized')
    
    def p_word_given_class(self, data_files, vocab_size):
        """
        helper function
        return dictionary representation of P(word | class)
        """
        word_counter = Counter()
        for file in data_files:
            words = get_words(open_file(file))
            for i, word in enumerate(words):
                if unigrams:
                    if word not in self.vocab:
                        continue
                    word_counter[word] += 1
                if bigrams:
                    if i != 0:
                        bigram = words[i - 1] + ' ' + word
                        if bigram not in self.vocab:
                            continue
                        word_counter[bigram] += 1
        word_counter["UNK"] = 0
        total_count = sum(word_counter.values())
        for word in word_counter:
            word_counter[word] = (word_counter[word] + self.k)/ (total_count + vocab_size * self.k)
        p_word_given_class = dict(word_counter)
        return p_word_given_class

    def p_class_given_input(self, input_path, p_word_given_class, p_class):
        """
        helper function
        return P(class | input)
        """
        p_class_given_input = 0
        words = get_words(open_file(input_path))
        if bigrams:
            sample_bigrams = set()
            for i, word in enumerate(words):
                if i != 0:
                    sample_bigrams.add(words[i-1] + ' ' + word)
            for bigram in sample_bigrams:
                if bigram not in self.vocab:
                    continue
                if bigram not in p_word_given_class:
                    bigram = "UNK"
                p = p_word_given_class[bigram]
                if p > 0:
                    p_class_given_input += math.log(p_word_given_class[bigram])
        if unigrams:
            for word in set(words):
                if word not in self.vocab:
                    continue
                if word not in p_word_given_class:
                    word = "UNK"
                p = p_word_given_class[word]
                if p > 0:
                    p_class_given_input += math.log(p_word_given_class[word])
        p_class_given_input += math.log(p_class)
        return p_class_given_input
        
    def train(self, X_train_pos, X_train_neg):
        #print('computing P(word|class)')
        # compute P(word | positive) and P(word | negative)
        self.p_word_given_pos = self.p_word_given_class(X_train_pos, self.vocab_size)
        self.p_word_given_neg = self.p_word_given_class(X_train_neg, self.vocab_size)
        # compute P(positive) and P(negative_
        #print('computing P(class)')
        self.p_pos = (len(X_train_pos) + self.k) / (len(X_train_pos) + len(X_train_neg) + len(self.vocab) * self.k)    
        self.p_neg = (len(X_train_neg) + self.k) / (len(X_train_neg) + len(X_train_pos) + len(self.vocab) * self.k)
        #print('trained')

    def predict(self, X_sample):
        p_pos = self.p_class_given_input(X_sample, self.p_word_given_pos, self.p_pos)        
        p_neg = self.p_class_given_input(X_sample, self.p_word_given_neg, self.p_neg)
        if p_pos > p_neg:
            return 'pos'
        else:
            return 'neg'
        
    def evaluate(self, X, ground_truth):
        count_correct = 0
        for sample in X:
            if self.predict(sample) == ground_truth:
                count_correct += 1
        return float(count_correct)/len(X)




