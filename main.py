import json
import utils
from nb import NaiveBayesProductRater

if __name__ == '__main__':

    print('initializing...')
    nb = NaiveBayesProductRater()
    nb.grams = 1
    
    print('training...')
    nb.train(X_train_pos, X_train_neg)

    print('testing...')
    pos_acc = nb.evaluate(X_test_pos, 'pos')
    neg_acc = nb.evaluate(X_test_neg, 'neg')
        
    print('results:\npositive ratings: {0:.3f}\nnegative ratings: {0:.3f}'.format(pos_acc, neg_acc)
    json.dump([pos_acc, neg_acc], open('test_accuracies.json', 'w'))