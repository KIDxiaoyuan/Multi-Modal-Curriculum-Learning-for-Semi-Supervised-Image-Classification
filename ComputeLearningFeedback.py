import numpy as np


def ComputeLearningFeedback(F_Curriculum, Gamma):
    # Compute learning feedback via Eq.17
    # Entropy of labels
    CurriculumTotal, ClassTotal = np.shape(F_Curriculum)
    SumEntropy = np.zeros((CurriculumTotal, 1))
    for i in range(0, ClassTotal):
        Temp = F_Curriculum[:, i] * (np.log(F_Curriculum[:, i]) / np.log(ClassTotal))
        SumEntropy = SumEntropy - Temp.reshape(-1, 1)

    SumEntropy[np.isnan(SumEntropy)] = 0
    AverageEntropy = np.mean(SumEntropy, axis=0)
    LearningEval = np.exp(-Gamma * AverageEntropy)
    return LearningEval[0]
