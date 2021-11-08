import numpy as np
from scipy import sparse as sps

"""
% Conduct label propagation via Eq.13
% Input:
% Curriculum: Indices of curriculum examples
% LabeledIndex: Indices of currently labeled examples
% InitialLabeledIndex: Indices of initial labeled examples
% P: Iteration matrix
% F: label matrix

% Output:
% F_New: the updated label matrix
"""


def LabelPropagation(Curriculum, LabeledIndex, InitialLabeledIndex, P, F):
    DataTotal, ClassTotal = np.shape(F)
    # find soft labels
    IndexForCurrentPropgation = np.concatenate((LabeledIndex.reshape(-1, 1), Curriculum.reshape(-1, 1)), axis=0)
    M = np.zeros((DataTotal, DataTotal))
    M[0, IndexForCurrentPropgation.astype(int)] = 1
    M = sps.csr_matrix(np.diag(np.sum(M, axis=0)))

    F_New = M.dot(np.transpose(P))
    F_New = F_New.dot(F)

    F_New[InitialLabeledIndex, :] = F[InitialLabeledIndex, :]  # Clamp the labeled examples
    F_New[np.sum(F_New, axis=1) == 0, :] = 1 / ClassTotal

    F_New = np.divide(F_New, np.sum(F_New, axis=1).reshape(-1, 1))
    F_New[np.isnan(F_New)] = 0
    return F_New
