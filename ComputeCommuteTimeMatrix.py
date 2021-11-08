import numpy as np


def ComputeCommuteTimeMatrix(L):
    # Compute commute time matrix
    n = L.shape[0]
    CT = np.zeros(L.shape)
    L_Inverse = np.linalg.pinv(L)
    for i in range(0, n):
        for j in range(i + 1, n):
            CT[i, j] = L_Inverse[i, i] + L_Inverse[j, j] - 2 * L_Inverse[i, j]
            CT[j, i] = CT[i, j]
    return CT
