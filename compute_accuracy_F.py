import numpy as np


def compute_accuracy_F(actual, pred):
    """
     GETCM : gets confusion matrices, precision, recall, and F scores
% [confus,numcorrect,precision,recall,F] = getcm (actual,pred,[classes])
%
% actual is a N-element vector representing the actual classes
% pred is a N-element vector representing the predicted classes
% classes is a vector with the numbers of the classes (by default, it is 1:k, where k is the
%    largest integer to appear in actual or pred.
%
% dinoj@cs.uchicago.edu , Apr 2005, modified July 2005

    """
    if actual.shape[0] != pred.shape[0]:
        pred = np.transpose(pred)

    numcorrect = np.sum(actual == pred, axis=0)
    accuracy = numcorrect / actual.shape[0]
    return accuracy
