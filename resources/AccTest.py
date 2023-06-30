import numpy as np

class AccTest:
    def __init__(self, outN, labels):
        OutMaxArg = np.argmax(outN, axis=1)
        LabelMaxArg = np.argmax(labels, axis=1)
        self.accuracy = np.mean(OutMaxArg == LabelMaxArg)