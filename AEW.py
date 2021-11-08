"""
Adaptive Edge Weighting for Label Propagation
%
% INPUT
%  X: d (features) times n (instances) input data matrix
%  param: The structure variable containing the following field:
%    max_iter: The maximum number of iteration for gradient descent
%    k: The number of nearest neighbors
%    sigma: Initial width parameter setting 'median'|'local-scaling'
% OUTPUT
%  W: The optimized weighted adjacency matrix
%  W0: The initial adjacency matrix
% REFERENCE
%  M. Karasuyama and H. Mamitsuka, "Manifold-based similarity
%  adaptation for label propagation", NIPS 2013.
"""
import numpy as np
from scipy.spatial.distance import squareform, pdist

from generate_nngraph import generate_nngraph


def diag(x):
    if x.shape[0] == 1:
        return np.diag(x.reshape(-1))
    else:
        return np.diag(x)


def my_length(sigma0):
    if type(sigma0) is np.ndarray:
        return sigma0.shape[0]
    else:
        return 1


def AEW(X, AEW_K, AEW_sigma, AEW_max_iter):
    (d, n) = np.shape(X)
    ex = 0
    tol = 1e-4
    # 行搜索的参数Parameters for line-search
    beta = 0.1
    beta_p = 0
    max_beta_p = 8
    rho = 1e-3
    W0, sigma0 = generate_nngraph(X, AEW_K, AEW_sigma)
    L = np.identity(d)
    Xori = X;
    if my_length(sigma0) > 1:
        Dist = squareform(pdist(np.transpose(X)) ** 2)
        sigma0 = sigma0.reshape(n, 1)
        Dist = np.divide(Dist, np.dot(sigma0, np.transpose(sigma0)))
    else:
        X = np.divide(X, np.dot(np.sqrt(2), sigma0))
        Dist = squareform(pdist(np.transpose(X)) ** 2)
    edge_idx = np.where(W0)
    W = np.zeros((n, n))
    W[edge_idx] = np.exp(-Dist[edge_idx])
    Gd = np.zeros((n, n, d))
    W_idx = np.empty(n, dtype=object)
    for i in range(0, n):
        W_idx[i] = np.where(W[i, :])
        for j in W_idx[i][0]:
            if W[i, j]:
                Gd[i, j, :] = -((X[:, i] - X[:, j]) * (X[:, i] - X[:, j]))
                if my_length(sigma0) > 1:
                    Gd[i, j, :] = np.divide(Gd[i, j, :], np.dot(sigma0[i], sigma0[j]))

    """
    Gradient Descent
    """
    d_W = np.zeros((n, n, d))
    d_WDi = np.zeros((n, n, d))
    sum_d_W = np.zeros((n, d))
    for Iter in range(0, AEW_max_iter):
        D = W.sum(axis=0)
        for i in range(0, n):
            d_W[i, W_idx[i][0], :] = np.dot(np.dot(2, diag(W[i, W_idx[i][0]])),
                                            Gd[i, W_idx[i][0], :].reshape(W_idx[i][0].shape[0], d) *
                                            (np.dot(np.ones((W_idx[i][0].shape[0], 1)),
                                                    np.transpose(diag(L).reshape(-1, 1)))))
        for i in range(0, n):
            sum_d_W[i, :] = np.sum(d_W[i, W_idx[i][0], :], axis=0)
            eq1 = np.divide(d_W[i, W_idx[i][0], :], D[i])
            eq2 = np.divide(np.transpose(W[i, W_idx[i][0]]), D[i] ** 2)
            eq2 = eq2.reshape(-1, 1)
            eq3 = sum_d_W[i, :].reshape(1, d)
            eq4 = np.dot(eq2, eq3)
            d_WDi[i, W_idx[i][0], :] = eq1 - (eq4.reshape(1, W_idx[i][0].shape[0], d))
        Xest = np.transpose(np.dot(np.dot(diag(np.divide(1, D)), W), np.transpose(Xori)))
        err = Xori - Xest
        sqerr = np.sum(np.sum(err ** 2, axis=0), axis=0)
        grad = -np.transpose(
            np.dot(np.transpose(d_WDi.reshape(n ** 2, d)), np.dot(np.transpose(err), Xori).reshape(-1, 1)))
        grad = np.divide(grad, np.linalg.norm(grad))  # normalize
        print("Iter = ", Iter, " MSE = ", sqerr / (d * n), "\n")
        step = (beta ** beta_p) * 1
        sqerr_prev = sqerr
        L_prev = L
        while 1:  # line search
            L = L_prev - np.dot(step, diag(grad))
            dist = squareform(pdist(np.transpose(np.dot(L, X))) ** 2)
            if my_length(sigma0) > 1:
                dist = np.divide(dist, np.dot(sigma0, np.transpose(sigma0)))
            W[edge_idx] = np.exp(-dist[edge_idx])

            D = np.sum(W, axis=0)
            Xest = np.transpose(np.dot(np.dot(diag(np.divide(1, D)), W), np.transpose(Xori)))
            err = Xori - Xest
            sqerr_temp = np.sum(np.sum(err ** 2, axis=0), axis=0)
            if np.less_equal(sqerr_temp - sqerr_prev,
                             -np.dot(np.dot(np.dot(rho, step), np.transpose(grad)), grad)).all():
                break

            beta_p = beta_p + 1
            if beta_p > max_beta_p:
                ex = 1
                break
            step = step * beta
        if ((sqerr_prev - sqerr_temp) / sqerr_prev) < tol or ex:
            break

    return W
