import numpy as np


def UpdateInvKernelLabeledMat(InvKernelLabeledMat, KernelMat, LabeledIndex, Curriculum):
    LabeledTotal = LabeledIndex.shape[0]
    CurriculumTotal = Curriculum.shape[0]
    UpdatedLabeledTotal = LabeledTotal + CurriculumTotal
    # Blockwise inversion update
    B = KernelMat[LabeledIndex.reshape(-1, 1).astype(int), Curriculum.reshape(1, -1).astype(int)]
    C = KernelMat[Curriculum.reshape(-1, 1).astype(int), LabeledIndex.reshape(1, -1).astype(int)]
    D = KernelMat[Curriculum.reshape(-1, 1).astype(int), Curriculum.reshape(1, -1).astype(int)]
    invA = InvKernelLabeledMat
    tempInverse = np.linalg.solve(D - np.dot(np.dot(C, invA), B), np.eye(Curriculum.shape[0]))
    InvKernelLabeledMat_11 = invA + invA @ B @ tempInverse @ C @ invA
    InvKernelLabeledMat_12 = -invA @ B @ tempInverse
    InvKernelLabeledMat_21 = np.transpose(InvKernelLabeledMat_12)
    InvKernelLabeledMat_22 = tempInverse

    # InvKernelLabeledMat = [InvKernelLabeledMat_11 InvKernelLabeledMat_12;InvKernelLabeledMat_21 InvKernelLabeledMat_22]
    InvKernelLabeledMat = np.zeros((UpdatedLabeledTotal, UpdatedLabeledTotal))
    InvKernelLabeledMat[0: LabeledTotal, 0: LabeledTotal] = InvKernelLabeledMat_11
    InvKernelLabeledMat[0: LabeledTotal, LabeledTotal:] = InvKernelLabeledMat_12
    InvKernelLabeledMat[LabeledTotal:, 0: LabeledTotal] = InvKernelLabeledMat_21
    InvKernelLabeledMat[LabeledTotal:, LabeledTotal:] = InvKernelLabeledMat_22
    InvKernelLabeledMat = np.dot(0.5, (InvKernelLabeledMat + np.transpose(InvKernelLabeledMat)))

    # permutation
    PermutedIndex = np.concatenate((LabeledIndex.reshape(-1, 1), Curriculum.reshape(-1, 1)), axis=0)
    OrderedIndex = np.sort(PermutedIndex, axis=0)
    II = np.argsort(PermutedIndex, axis=0)

    PermutationMatTranspose = np.zeros((UpdatedLabeledTotal, UpdatedLabeledTotal))
    ElementOneIndex = sub2ind([UpdatedLabeledTotal, UpdatedLabeledTotal], np.arange(UpdatedLabeledTotal).reshape(-1, 1),
                              II)
    PermutationMatTranspose = PermutationMatTranspose.reshape(-1, 1)
    PermutationMatTranspose[ElementOneIndex.astype(int)] = 1
    PermutationMatTranspose = np.reshape(PermutationMatTranspose, (UpdatedLabeledTotal, UpdatedLabeledTotal), order='F')
    InvKernelLabeledMat = PermutationMatTranspose @ InvKernelLabeledMat @ np.transpose(PermutationMatTranspose)
    return InvKernelLabeledMat


def sub2ind(array_shape, rows, cols):
    ind = np.zeros(rows.shape)
    for i in range(0, rows.shape[0]):
        ind[i] = rows[i] + cols[i] * array_shape[0]
    return ind
