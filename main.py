import json
import utils
import numpy as np
from nb import NaiveBayesProductRater

numfolds = 10

if __name__ == '__main__':

    results = []
    for fold in range (numfolds):
        #print('\nINITIALIZING...')
        nb = NaiveBayesProductRater(fold)

        #print('\nTRAINING...')
        nb.train(nb.X_train_pos, nb.X_train_neg)

        #print('\nTESTING...')
        pos_acc = nb.evaluate(nb.X_test_pos, 'pos')
        neg_acc = nb.evaluate(nb.X_test_neg, 'neg')
            
        #print('positive ratings: {0:.3f}\nnegative ratings: {1:.3f}\n'.format(pos_acc, neg_acc))
        totalacc = (pos_acc + neg_acc) / 2
        results.append(totalacc)
        print('fold {0} accuracy: {1:.3f}'.format(fold, totalacc))

    mean = sum(results) / len(results)        
    variance = sum((xi - mean) ** 2 for xi in results) / len(results)
    print('mean accuracy: {0:.3f}\nvariance: {1:.3f}'.format(mean, variance))
    
    

# smoothed
# fold 0 accuracy: 0.805
# fold 1 accuracy: 0.865
# fold 2 accuracy: 0.820
# fold 3 accuracy: 0.905
# fold 4 accuracy: 0.825
# fold 5 accuracy: 0.875
# fold 6 accuracy: 0.845
# fold 7 accuracy: 0.835
# fold 8 accuracy: 0.855
# fold 9 accuracy: 0.880
# mean accuracy: 0.851
# variance: 0.001

# unsmoothed
# fold 0 accuracy: 0.655
# fold 1 accuracy: 0.680
# fold 2 accuracy: 0.665
# fold 3 accuracy: 0.705
# fold 4 accuracy: 0.585
# fold 5 accuracy: 0.675
# fold 6 accuracy: 0.655
# fold 7 accuracy: 0.670
# fold 8 accuracy: 0.650
# fold 9 accuracy: 0.685
# mean accuracy: 0.662
# variance: 0.001
