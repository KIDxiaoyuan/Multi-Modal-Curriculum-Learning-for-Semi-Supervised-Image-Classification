import numpy as np


def ComputeLabelFusionWeights(ReconError1, ReconError2):
    # compute fusion weight via Eq.16
    Beta = 1
    View1 = np.exp(-Beta * ReconError1)
    View2 = np.exp(-Beta * ReconError2)

    Denominator = View1 + View2

    weight1 = View1 / Denominator
    weight2 = View2 / Denominator
    return weight1, weight2
