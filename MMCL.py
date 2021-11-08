import numpy as np
import scipy.io
from scipy import sparse as sps
import faiss
from AEW import AEW
from ComputeCommuteTimeMatrix import ComputeCommuteTimeMatrix
from ComputeLabelFusionWeights import ComputeLabelFusionWeights
from ComputeLearningFeedback import ComputeLearningFeedback
from DecideCurriculum import DecideCurriculum
from LabelPropagation import LabelPropagation
from UpdateInvKernelLabeledMat import UpdateInvKernelLabeledMat


def MMCL(Feature_1, Feature_2, GroundTruth, LabeledIndex, UnlabeledIndex):
    # 参数设置
    Alpha = 1  # the trade-off parameter in Eq.8
    Gamma = 2  # Learning rate in Eq.17
    Kappa = 1.1  # Gamma := Gamma/Kappa in every propagation
    Theta = 0.05  # theta in Eq.20

    # 通过AEW构建图像
    AEW_K = 10  # The number of neighbors in kNN graph
    AEW_sigma = 'local-scaling'  # Kernel parameter heuristics 'median' or 'local-scaling'核参数启发式“中值”或“局部缩放”
    AEW_max_iter = 100

    # 导入数据
    DataTotal = Feature_1.shape[1]

    ClassTotal = np.max(GroundTruth).astype(int) + 1

    W1 = AEW(Feature_1, AEW_K, AEW_sigma, AEW_max_iter)
    W2 = AEW(Feature_2, AEW_K, AEW_sigma, AEW_max_iter)

    KNNMat1 = np.zeros((DataTotal, DataTotal))
    KNNMat2 = np.zeros((DataTotal, DataTotal))

    KNNMat1[W1 != 0] = 1
    KNNMat2[W2 != 0] = 1

    # Compute Laplacian Matrix and Kernel Matrix计算拉普拉斯矩阵和核矩阵
    D1 = sps.csr_matrix(np.diag(np.sum(W1, axis=0)))
    D2 = sps.csr_matrix(np.diag(np.sum(W2, axis=0)))
    L1 = D1 - W1
    L2 = D2 - W2
    I_DataTotal = np.eye(DataTotal)
    KernelMat1 = np.linalg.solve(L1 + np.dot(0.01, I_DataTotal), I_DataTotal)
    KernelMat1 = np.dot(0.5, (KernelMat1 + np.transpose(KernelMat1)))
    KernelMat2 = np.linalg.solve(L2 + np.dot(0.01, I_DataTotal), I_DataTotal)
    KernelMat2 = np.dot(0.5, (KernelMat2 + np.transpose(KernelMat2)))
    # Compute Transition Matrix
    P1 = np.divide(W1, np.sum(W1, axis=1))
    P1[np.isnan(P1)] = 0
    P2 = np.divide(W2, np.sum(W2, axis=1))
    P2[np.isnan(P2)] = 0
    # Compute Commute Time Matrix
    CT1 = ComputeCommuteTimeMatrix(L1)
    CT2 = ComputeCommuteTimeMatrix(L2)
    print("Graph Construction Completed!")

    # Y: initially labeled examples 初始化已经标记的示例
    Y = np.zeros((DataTotal, ClassTotal))

    for i in range(0, LabeledIndex.shape[0]):
        Y[LabeledIndex[i], GroundTruth[LabeledIndex[i]]] = 1
    # Label Propagation via Curruculum Learning 通过课程学习进行标签传播

    Iteration = 1
    F = Y
    InitialLabeledIndex = LabeledIndex
    LearningEval = 0.01
    InvKernelLabeledMat1 = np.linalg.solve(KernelMat1[LabeledIndex.reshape(-1, 1), LabeledIndex.reshape(1, -1)],
                                           np.eye(LabeledIndex.shape[0]))
    InvKernelLabeledMat1 = (InvKernelLabeledMat1 + np.transpose(InvKernelLabeledMat1)) / 2
    InvKernelLabeledMat2 = np.linalg.solve(KernelMat2[LabeledIndex.reshape(-1, 1), LabeledIndex.reshape(1, -1)],
                                           np.eye(LabeledIndex.shape[0]))
    InvKernelLabeledMat2 = (InvKernelLabeledMat2 + np.transpose(InvKernelLabeledMat2)) / 2
    while 1:
        # Decide curriculum for this iteration决定本次迭代的课程
        Curriculum, ReconError1, ReconError2 = DecideCurriculum(LabeledIndex, UnlabeledIndex, Y, LearningEval, Alpha,
                                                                KNNMat1, KNNMat2, KernelMat1, KernelMat2,
                                                                InvKernelLabeledMat1, InvKernelLabeledMat2,
                                                                CT1, CT2)
        # Label propagation & fusion标签传播和融合
        F1 = LabelPropagation(Curriculum, LabeledIndex, InitialLabeledIndex, P1, F)
        F2 = LabelPropagation(Curriculum, LabeledIndex, InitialLabeledIndex, P2, F)

        weight1, weight2 = ComputeLabelFusionWeights(ReconError1, ReconError2)
        F = np.dot(weight1, F1) + np.dot(weight2, F2)

        # Evaluate the learning performance of this learning round 评估本轮学习表现
        F_Curriculum = F[Curriculum.reshape(-1).astype(int), :]
        LearningEval = ComputeLearningFeedback(F_Curriculum, Gamma)

        # find Y for next loop
        Y = np.zeros((DataTotal, ClassTotal))  # The labels of learned curriculum
        Y_temp = np.argmax(F, axis=1)
        IndexForCurrentPropgation = np.concatenate((LabeledIndex.reshape(-1, 1), Curriculum.reshape(-1, 1)), axis=0)
        Y_temp2 = Y_temp[IndexForCurrentPropgation.astype(int)]
        Y.reshape(-1, 1)[sub2ind(Y.shape, IndexForCurrentPropgation, Y_temp2).astype(int)] = 1
        Y = Y.reshape(-1, 1)
        Y = Y.reshape(DataTotal, ClassTotal, order='F')
        # Update variables incrementally 增量更新变量
        InvKernelLabeledMat1 = UpdateInvKernelLabeledMat(InvKernelLabeledMat1, KernelMat1, LabeledIndex,
                                                         Curriculum)  # update inv_K_LL
        InvKernelLabeledMat2 = UpdateInvKernelLabeledMat(InvKernelLabeledMat2, KernelMat2, LabeledIndex,
                                                         Curriculum)  # update inv_K_LL

        Gamma = Gamma / Kappa
        LabeledIndex = np.concatenate((LabeledIndex.reshape(-1, 1), Curriculum.reshape(-1, 1)), axis=0)
        LabeledIndex = np.sort(LabeledIndex, axis=0)
        if LabeledIndex.shape[0] == DataTotal:
            break
        else:
            AllIndex = np.transpose(np.arange(DataTotal))
            AllIndex = np.delete(AllIndex, LabeledIndex.astype(int), axis=0)
            UnlabeledIndex = AllIndex
            Iteration = Iteration + 1
    # Iterate Untill Convergence 迭代直到收敛

    F1 = np.linalg.solve(np.eye(DataTotal) - np.dot(Theta, P1), F)
    F2 = np.linalg.solve(np.eye(DataTotal) - np.dot(Theta, P2), F)
    F = F1 + F2 / 2

    # Output
    Classification = np.argmax(F, axis=1)
    return Classification


def sub2ind(array_shape, rows, cols):
    ind = np.zeros(rows.shape)
    for i in range(0, rows.shape[0]):
        ind[i] = rows[i] + cols[i] * array_shape[0]
    return ind


def maxminnorm(array):
    maxcols = array.max(axis=0)
    mincols = array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t = np.empty((data_rows, data_cols))
    for i in range(data_cols):
        t[:, i] = (array[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t
