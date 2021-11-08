"""
Create nearest-neighbor graph
%
% Input:
%  X: Input data
%  k: The number of nearest-neighbor
%  sigma: Heauristics for setting Gaussian width 'median' or 'local-scaling'
"""
import faiss
import numpy as np
from scipy.spatial.distance import squareform, pdist


def generate_nngraph(X, k, sigma):
    D = squareform(pdist(np.transpose(X)) ** 2)
    sort_idx = np.transpose(np.argsort(D))
    sort_D = D
    sort_D.sort(axis=0)
    n = np.shape(X)[1]
    if np.any(np.not_equal(np.transpose(sort_idx[0, :]), np.transpose(np.arange(0, n)))):
        temp_idx = np.where(np.not_equal(sort_idx[0, :], range(0, n)))
        (I, J) = np.where(np.equal(sort_idx[:, temp_idx], np.dot(np.ones(n, 0), temp_idx)))
        if np.size(I, axis=1) != np.size(temp_idx, axis=1):
            print("error in generate_nngraph.py line 28")
        for i in range(0, np.size(I, axis=1)):
            temp = sort_idx[I[i], temp_idx[i]]
            sort_idx[I[i], temp_idx[i]] = sort_idx[0, temp_idx[i]]
            sort_idx[0, temp_idx[i]] = temp

    knn_idx = sort_idx[1:k + 1, :]
    kD = sort_D[1:k + 1, :]
    W = np.zeros((n, n))
    if sigma == 'median':
        sigma = np.sqrt(kD).mean()
        if sigma == 0:
            sigma = 1
        for i in range(0, n):
            W[i, knn_idx[:, i]] = np.exp(np.divide(-kD[:, i], (2 * sigma * sigma)))
    elif sigma == 'local-scaling':
        if k < 7:
            sigma = np.sqrt(kD[-1, :])
        else:
            sigma = np.sqrt(kD[6, :])
        sigma[sigma == 0] = 1
        for i in range(0, n):
            W[i, knn_idx[:, i]] = np.exp(
                np.divide(-kD[:, i], np.dot(
                    sigma[i], np.transpose(sigma[knn_idx[:, i]]))))
    else:
        print("'Unknown option for sigma'")
    W = np.maximum(W, np.transpose(W))
    return W, sigma
