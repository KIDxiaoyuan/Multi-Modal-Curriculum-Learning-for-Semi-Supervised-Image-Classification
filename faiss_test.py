import faiss
import numpy
import numpy as np
import scipy.io


Feature_1 = scipy.io.loadmat('image_feature2w.mat')
Feature_1 = Feature_1["image_feature"]
Feature_1 = numpy.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 3.], [1., 2., 3.]]).astype('float32')
#
d = Feature_1.shape[1]
nb = Feature_1.shape[0]
# #
# # print(d)
# index = faiss.IndexFlatL2(d)
index = faiss.IndexFlatIP(d)
# # print(d)
# index.add(np.ascontiguousarray(Feature_1))
# index.add(Feature_1)
# print(index.ntotal)
# AVA_DatasetSplitIdx = scipy.io.loadmat('AVA_DatasetSplitIdx2w.mat')
# LabeledIndex = AVA_DatasetSplitIdx["LabeledIndex"].reshape(-1) - 1
# UnlabeledIndex = AVA_DatasetSplitIdx["UnlabeledIndex"].reshape(-1) - 1
# Unlabeled = Feature_1[UnlabeledIndex]
# Labeled = Feature_1[LabeledIndex]
# print(LabeledIndex[:5])
# print(Labeled)
k = 2
D, I = index.search(Feature_1, k)
# print(I.shape)
print(D[:2])
print(I[:2])
