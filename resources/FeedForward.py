import numpy as np

class FeedForward:
    def __init__(self, samples, Wh, bh, Wo, bo):
        self.OutH1 = np.sign(np.dot(samples, Wh.T) + bh)
        self.OutN = np.sign(np.dot(self.OutH1, Wo.T) + bo)