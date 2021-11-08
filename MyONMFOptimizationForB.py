import numpy as np
import scipy.linalg


def MyONMFOptimizationForB(R, B, B_Star, Alpha, Sigma):
    # User Settings
    C = 1.2
    minOpts_maxMinIters = 1
    lsOpts_stepBeta = 1.1
    lsOpts_stepLowerThreshold = 1e-15

    SigmaUpperThreshold = 1e10
    tol = 1e-4
    MaxIter = 100
    # =====================================
    Lembda = np.zeros(B.shape)
    stepB = 1
    error = np.zeros(MaxIter)
    converged = False
    Iter = 0
    T = np.maximum(0, B + Lembda / Sigma)
    while not converged:
        Iter = Iter + 1
        # update B
        B_New, stepB, Fvalue = updateB(B, B_Star, R, T, Lembda, Alpha, Sigma, stepB, minOpts_maxMinIters,
                                       lsOpts_stepBeta, lsOpts_stepLowerThreshold)
        # stop Criterion
        error[Iter] = np.linalg.norm(B_New - B, ord='fro')
        if ((error[Iter] / np.linalg.norm(B, ord='fro')) < tol) or (Iter == MaxIter):
            converged = True
        # update parameters
        Lembda = np.maximum(0, Lembda - np.dot(Sigma, B_New))
        Sigma = np.minimum(np.dot(Sigma, C), SigmaUpperThreshold)
        T = np.maximum(0, B_New + Lembda / Sigma)
        B = B_New
    return B


def updateB(B, B_Star, R, T, Lembda, Alpha, Sigma, step, minOpts_maxMinIters, lsOpts_stepBeta,
            lsOpts_stepLowerThreshold):
    agrad = lambda B: np.dot(Sigma, B - T) + Lembda + (np.dot(np.dot(2, R), B)) + np.dot(np.dot(2, Alpha), B - B_Star)
    stepMove = lambda B, step: stepMoveB(B, step, agrad)
    ObjFunValue = lambda B: ComputeLagrangianObjFunValue(B, B_Star, R, T, Alpha, Sigma, Lembda)
    return minimizeFun(B, step, stepMove, ObjFunValue, minOpts_maxMinIters, lsOpts_stepBeta, lsOpts_stepLowerThreshold)


def stepMoveB(B, step, agrad):
    B_euc = B - np.dot(step, agrad(B))
    B_new = closestOrthogonalMatrix(B_euc)
    return B_new


def closestOrthogonalMatrix(A):
    # computes B s.t. A = B*P, B'*B = I
    P = scipy.linalg.sqrtm(np.dot(np.transpose(A), A))
    B = np.dot(A, np.linalg.inv(P))
    return B


def ComputeLagrangianObjFunValue(B, B_Star, R, T, Alpha, Sigma, Lembda):
    BMinusT = B - T
    G1 = np.dot(np.dot(2, R), B)
    G2 = Lembda
    G3 = np.dot(Sigma, BMinusT)
    G4 = np.dot(Alpha, (B - B_Star))

    # %G = G1+G2+G3;
    # % F = trace(X'*R*X + Lembda'*(X-T) + Sigma*((X-T)'*(X-T))/2);
    F = np.sum(np.dot(0.5, np.einsum('ij,ij->j', G1, B)) + np.einsum('ij,ij->j', G2, BMinusT) + np.dot(0.5, np.einsum(
        'ij,ij->j', G3, BMinusT)) + np.einsum('ij,ij->j', G4, B - B_Star), axis=0)
    return F


def minimizeFun(x, step, stepMove, ObjFunValue, minOpts_maxMinIters, lsOpts_stepBeta, lsOpts_stepLowerThreshold):
    lagrValue, lagrVal10 = 0.0, 0.0
    tolFun = 1e-4
    diffFun = np.inf  # difference in subsequent function values
    Iter = 1
    while (diffFun > tolFun) and (Iter <= minOpts_maxMinIters):
        x, step, lagrValue, lagrVal10 = lineSearch(x, step, stepMove, ObjFunValue, lsOpts_stepBeta,
                                                   lsOpts_stepLowerThreshold)
        diffFun = np.abs(lagrVal10 - lagrValue)
        Iter = Iter + 1
    # outer iter
    return x, step, lagrValue


def lineSearch(x, step, stepMove, computeLagrangianValueForB, lsOpts_stepBeta, lsOpts_stepLowerThreshold):
    startLagrangValue = computeLagrangianValueForB(x)
    lastLagrangVal = startLagrangValue
    isStepAccepted = False
    j = 1

    while (not isStepAccepted) and (step > lsOpts_stepLowerThreshold):
        x_new = stepMove(x, step)
        lagrVal_candidate = computeLagrangianValueForB(x_new)
        hasImproved = (lagrVal_candidate < startLagrangValue) and (lagrVal_candidate < lastLagrangVal)
        if j == 1:
            keepIncreasing = hasImproved
            last_x = x_new
        if keepIncreasing:
            if hasImproved:
                last_x = x_new
                lastLagrangVal = lagrVal_candidate
                step = lsOpts_stepBeta * step
            else:
                step = step / lsOpts_stepBeta
                x = last_x
                isStepAccepted = True
        else:
            if hasImproved:
                lastLagrangVal = lagrVal_candidate
                x = x_new
                isStepAccepted = True
            else:
                step = step / lsOpts_stepBeta
        j = j + 1
    lagrValue = lastLagrangVal
    return x, step, lagrValue, startLagrangValue
