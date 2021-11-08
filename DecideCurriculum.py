import numpy as np
from scipy import sparse as sps

from MyONMFOptimizationForB import MyONMFOptimizationForB
from MyONMFOptimizationForBStar import MyONMFOptimizationForBStar


def DecideCurriculum(LabeledIndex, UnlabeledIndex, Y, LearningEval, Alpha, KNNMat1, KNNMat2, KernelMat1, KernelMat2,
                     InvKernelLabeledMat1, InvKernelLabeledMat2, CT1, CT2):
    """
% This function decides the curriculum for current learning round
% Input:
% LabeledIndex: the indices of labeled examples
% UnlabeledIndex: the indices of unlabeled examples
% Y: binary label matrix, each row represents an example二进制标签矩阵，每行代表一个示例
% LearningEval: Conf(.) in Eq.18
% Alpha: The trade-off parameter in Eq.8
% KNNMat: the n*n matrix encoding the neighborhood information
% KernelMat: Kernel matrix
% InvKernelLabeledMat: inv(KernelMat);
% CT: commute time matrix

% Output:
% Curriculum: the index of decided examples for this loop
% ReconError: ||S^{(v)[t]}-S^{(*)[t]}|| in Eq.16
    """
    # global B_Star_Optimal

    UnlabeledTotal = UnlabeledIndex.shape[0]  # the total number of unlabeled examples

    # % --------------------------Find candidates of each view based on KNNMat---------%
    KNNMat1[LabeledIndex.reshape(-1, 1).astype(int), LabeledIndex.reshape(1, -1).astype(int)] = 0
    NNofLabeled1 = KNNMat1[LabeledIndex.reshape(-1).astype(int), :]
    CandidateIndex1 = np.transpose(np.where(np.sum(NNofLabeled1, axis=0) != 0))
    KNNMat2[LabeledIndex.reshape(-1, 1).astype(int), LabeledIndex.reshape(1, -1).astype(int)] = 0
    NNofLabeled2 = KNNMat2[LabeledIndex.reshape(-1).astype(int), :]
    CandidateIndex2 = np.transpose(np.where(np.sum(NNofLabeled2, axis=0) != 0))
    CandidateIndex = np.union1d(CandidateIndex1, CandidateIndex2)
    CandidateTotal = CandidateIndex.shape[0]

    # Decide the amount of Curriculum points
    CurriculumSize = np.ceil(CandidateTotal * LearningEval)

    # Prepare diagonal commute time matrix M_C
    ClassTotal = Y.shape[1]
    M1 = ComputeM(ClassTotal, CT1, CandidateIndex, CandidateTotal, Y)
    M2 = ComputeM(ClassTotal, CT2, CandidateIndex, CandidateTotal, Y)

    # --------------Decide Curriculum -----------------
    R1 = ComputeR(KernelMat1, InvKernelLabeledMat1, M1, CandidateIndex, LabeledIndex)
    R2 = ComputeR(KernelMat2, InvKernelLabeledMat2, M2, CandidateIndex, LabeledIndex)

    # --------------------Find B using optimization on Stiefel manifold-------------------
    CurriculumSize = CurriculumSize.astype(int)
    # CandidateTotal = CandidateTotal.astype(int)
    B1 = np.zeros((CandidateTotal, CurriculumSize))
    # CurriculumSize = CurriculumSize.astype(int)
    B1[0:CurriculumSize, :] = np.eye(CurriculumSize)
    B2 = B1
    B_Star = B1
    Sigma = 1  # initial value of penalty coefficient
    ObjFunValue = 1e8
    Converge = False
    Iter = 0
    Tol = 1e-4
    MaxIter = 50

    while not Converge:

        ObjFunValue_Old = ObjFunValue
        # Update View 1  View2
        B1 = MyONMFOptimizationForB(R1, B1, B_Star, Alpha, Sigma)
        B2 = MyONMFOptimizationForB(R2, B2, B_Star, Alpha, Sigma)

        # Update B_Star
        B_Star = MyONMFOptimizationForBStar(B1, B2, B_Star, Sigma)

        # Compute the value of objective function
        ObjFunValue, ReconError1, ReconError2 = ComputeObjFunValue(B1, B2, B_Star, R1, R2, Alpha)

        # Check termination
        ObjFunValueDiff = np.zeros(MaxIter)

        ObjFunValueDiff[Iter] = ObjFunValue_Old - ObjFunValue
        if (ObjFunValueDiff[Iter] / ObjFunValue_Old <= Tol) or (Iter == MaxIter - 1):
            # if  (norm(B_Star-B_Star_Old)/norm(B_Star_Old)<=tol) || (iter==MaxIter)
            Converge = True
            B_Star_Optimal = B_Star
        # display details
        print('IterTime', Iter, 'ObjFunValue', ObjFunValue, ' ObjFunValueDiff/ObjFunValue_Old',
              ObjFunValueDiff[Iter] / ObjFunValue_Old)
        Iter = Iter + 1
    # Discretization
    Curriculum = np.zeros((CurriculumSize, 1))
    for i in range(0, CurriculumSize):
        m, ind = np.max(B_Star_Optimal.reshape(-1)), np.argmax(B_Star_Optimal.reshape(-1, order='F'))
        if m == 0:
            break
        r, c = ind2sub([CandidateTotal, CurriculumSize], ind)
        Curriculum[i] = CandidateIndex[r]
        B_Star_Optimal[r, :] = -10000
    Curriculum = np.sort(Curriculum, axis=0)
    # Curriculum = np.delete(Curriculum,np.where(Curriculum==0),axis=0)
    # Curriculum[Curriculum==0] = []
    # -Decide curriculum -
    if UnlabeledTotal == 1:
        Curriculum = UnlabeledIndex
    return Curriculum, ReconError1, ReconError2


def ind2sub(array_shape, ind):
    Temp = np.arange(array_shape[0] * array_shape[1]).reshape(array_shape[1], array_shape[0])
    r, c = np.where(Temp == ind)
    return c, r


def ComputeM(ClassTotal, CT, CandidateIndex, CandidateTotal, Y):
    AverageCTtoClass = np.full((CandidateTotal, ClassTotal), np.nan)
    for i in range(0, ClassTotal):
        AverageCTtoClass[:, i] = np.mean(CT[CandidateIndex.reshape(-1, 1), np.where(Y[:, i] == 1)[0].reshape(1, -1)],
                                         axis=1)
    AverageCTtoClass_Sort = np.sort(-AverageCTtoClass, axis=1)
    AverageCTtoClass_Sort = -AverageCTtoClass_Sort
    M = sps.csr_matrix(np.diag(1 / (AverageCTtoClass_Sort[:, -2] - AverageCTtoClass_Sort[:, -1])))
    return M


def ComputeR(KernelMat, Inv_KernelMat_LL, M, CandidateIndex, LabeledIndex):
    CandidateIndex = CandidateIndex.astype(int)
    LabeledIndex = LabeledIndex.astype(int)
    KernelMatCC = KernelMat[CandidateIndex.reshape(-1, 1), CandidateIndex.reshape(1, -1)]
    KernelMatCL = KernelMat[CandidateIndex.reshape(-1, 1), LabeledIndex.reshape(1, -1)]
    KernelMatLC = KernelMat[LabeledIndex.reshape(-1, 1), CandidateIndex.reshape(1, -1)]
    R = KernelMatCC - np.dot(np.dot(KernelMatCL, Inv_KernelMat_LL), KernelMatLC) + M
    R = (R + np.transpose(R)) / 2
    R[R < 0] = 0
    return R


def ComputeObjFunValue(B1, B2, B_Star, R1, R2, Alpha):
    ReconError1 = np.sum(np.einsum('ij,ij->j', B_Star - B1, B_Star - B1), axis=0)
    ReconError2 = np.sum(np.einsum('ij,ij->j', B_Star - B2, B_Star - B2), axis=0)
    G1 = ReconError1 + ReconError2
    G2 = np.sum(np.einsum('ij,ij->j', np.dot(R1, B1), B1) + np.einsum('ij,ij->j', np.dot(R2, B2), B2))
    ObjFunValue = np.dot(Alpha, G1) + G2
    return ObjFunValue, ReconError1, ReconError2
