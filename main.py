import numpy
import numpy as np
import scipy.io
from scipy import sparse as sps
import faiss
import torch
import heapq
from collections import Counter
from compute_accuracy_F import compute_accuracy_F
from MMCL import MMCL


def main():
    # 导入数据
    Feature_1 = scipy.io.loadmat('image_feature.mat')  # 图像特征
    Feature_1 = np.matrix.transpose(Feature_1["image_feature"])  # D * N 512 * 3000

    Feature_2 = scipy.io.loadmat('text_feature.mat')  # 文本特征
    Feature_2 = np.matrix.transpose(Feature_2["text_feature"])

    GroundTruth = scipy.io.loadmat('AVA_GroundTruth.mat')
    GroundTruth = GroundTruth["GroundTruth"].reshape(-1) - 1  # N, 0/1

    AVA_DatasetSplitIdx = scipy.io.loadmat('AVA_DatasetSplitIdx.mat')
    LabeledIndex = AVA_DatasetSplitIdx["LabeledIndex"].reshape(-1) - 1
    UnlabeledIndex = AVA_DatasetSplitIdx["UnlabeledIndex"].reshape(-1) - 1
    Result = np.empty((Feature_1.shape[1]))
    Result[LabeledIndex] = GroundTruth[LabeledIndex]

    # Graph Construction构建图
    Feature_1 = maxminnorm(np.transpose(Feature_1))
    Feature_2 = maxminnorm(np.transpose(Feature_2))

    Unlabeled_F1 = Feature_1[UnlabeledIndex]
    Labeled_F1 = Feature_1[LabeledIndex]
    Unlabeled_F2 = Feature_2[UnlabeledIndex]
    Labeled_F2 = Feature_2[LabeledIndex]

    while(len(LabeledIndex) < Feature_1.shape[0]):
        # Use Faiss
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(torch.cuda.device_count()) - 1
        # Find 1000 Labeled Index
        FMap1 = faiss.GpuIndexFlatIP(res, Feature_1.shape[1], flat_config)
        FMap1.add(np.ascontiguousarray(Feature_1.astype(numpy.float32)))
        FMap2 = faiss.GpuIndexFlatIP(res, Feature_2.shape[1], flat_config)
        FMap2.add(np.ascontiguousarray(Feature_2.astype(numpy.float32)))
        k = 1024
        Cho_size = 100
        D1, I1 = FMap1.search(Labeled_F1.astype(numpy.float32), k)
        D2, I2 = FMap2.search(Labeled_F2.astype(numpy.float32), k)

        I1_Cho = np.zeros(LabeledIndex.shape[0])
        I2_Cho = np.zeros(LabeledIndex.shape[0])
        S = set(LabeledIndex)

        for i in range(len(I1)):
            count = 0
            for j in range(len(I1[i])):
                if I1[i][j] in S:
                    count = count + 1
            I1_Cho[i] = count
        for i in range(len(I2)):
            count = 0
            for j in range(len(I2[i])):
                if I1[i][j] in S:
                    count = count + 1
            I2_Cho[i] = count
        I_Cho = I1_Cho + I2_Cho
        Cho_Labeled = LabeledIndex[heapq.nlargest(Cho_size, range(len(I_Cho)), I_Cho.take)]
        # Find 1000 Unlabeled Index
        FMap1 = faiss.GpuIndexFlatIP(res, Unlabeled_F1.shape[1], flat_config)
        FMap2 = faiss.GpuIndexFlatIP(res, Unlabeled_F2.shape[1], flat_config)
        FMap1.add(np.ascontiguousarray(Unlabeled_F1.astype(numpy.float32)))
        FMap2.add(np.ascontiguousarray(Unlabeled_F2.astype(numpy.float32)))
        D1, I1 = FMap1.search(Feature_1[Cho_Labeled].astype(numpy.float32), k)
        D2, I2 = FMap2.search(Feature_2[Cho_Labeled].astype(numpy.float32), k)
        I = np.concatenate((I1.reshape(-1), I2.reshape(-1)), axis=0)
        # print(len(Counter(I).most_common(Cho_size)))
        if len(Counter(I).most_common(Cho_size)) >= Cho_size:
            Cho_Unlabeled = Counter(I).most_common(Cho_size)
        else:
            Cho_Unlabeled = Counter(I).most_common(len(Counter(I)))
        Cho_Unlabeled = [x[0] for x in Cho_Unlabeled]
        Cho_Unlabeled = UnlabeledIndex[Cho_Unlabeled]
        # CHO_U = np.empty((len(Cho_Unlabeled)))
        # for i in range(len(Cho_Unlabeled)):
        #     print(i, np.where((Feature_1 == Unlabeled_F1[Cho_Unlabeled[i]]).all(axis = 1))[0][0])
            # CHO_U[i] = np.where((Feature_1 == Unlabeled_F1[Cho_Unlabeled[i]]).all(axis = 1))[0][0]
        # Cho_Unlabeled = CHO_U.astype(int)
            # print("exchange")
        # Cho_Unlabeled = np.array(Cho_Unlabeled).astype(int).tolist()

        # Get their feature and GroundTruth
        Cho_F1 = Feature_1[np.concatenate((Cho_Labeled, Cho_Unlabeled), axis=0)]
        Cho_F2 = Feature_2[np.concatenate((Cho_Labeled, Cho_Unlabeled), axis=0)]
        List_Labeled = np.array(range(0, len(Cho_Labeled)))
        List_UnLabeled = np.array(range(len(Cho_Labeled), len(Cho_Labeled) + len(Cho_Unlabeled)))
        Cho_GroundTruth = np.empty((len(Cho_Labeled) + len(Cho_Unlabeled)))
        Cho_GroundTruth[List_Labeled] = Result[Cho_Labeled]
        # #Use MMCL to Fusion
        # Classification = MMCL(np.transpose(Cho_F1), np.transpose(Cho_F2), Cho_GroundTruth.astype(int), List_Labeled, List_UnLabeled)
        # Update
        Classification = np.ones((len(Cho_GroundTruth))).astype(int)
        Result[Cho_Unlabeled] = Classification[List_UnLabeled]
        LabeledIndex = np.append(LabeledIndex,Cho_Unlabeled, axis=0)
        # COUNT = len(UnlabeledIndex)
        # for item in Cho_Unlabeled:
            # print(np.where(UnlabeledIndex == item))
        # print(len(UnlabeledIndex),np.max(Cho_Unlabeled))
        # Delete = UnlabeledIndex[Cho_Unlabeled]
        # print(UnlabeledIndex)
        # print(Delete)
        # print()
        # print("before",len(UnlabeledIndex),len(Cho_Unlabeled))
        bef = len(UnlabeledIndex)
        print("real_learn",len(np.unique(Cho_Unlabeled)))
        UnlabeledIndex = np.setdiff1d(UnlabeledIndex, Cho_Unlabeled)
        print(bef - len(UnlabeledIndex))
        # print("after", len(UnlabeledIndex))
        # print(COUNT - len(UnlabeledIndex))

        # print(COUNT)
        # UnlabeledIndex = np.delete(UnlabeledIndex, Cho_Unlabeled, axis=0)
        Unlabeled_F1 = Feature_1[UnlabeledIndex]
        Labeled_F1 = Feature_1[LabeledIndex]
        Unlabeled_F2 = Feature_2[UnlabeledIndex]
        Labeled_F2 = Feature_2[LabeledIndex]
        print(len(LabeledIndex), len(UnlabeledIndex))
        print("Now Learned",len(LabeledIndex) / (len(LabeledIndex)+ len(UnlabeledIndex)) * 100,"%")

    #
    # Evaluation
    acc = compute_accuracy_F(GroundTruth, Result)
    print("acc = ", acc)


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


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
