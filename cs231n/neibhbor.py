import numpy as np
class NearestNeighbor:
    def __init__(self):
        self.Xtr = None
        self.ytr = None

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, Xtst):
        n_test = Xtst.shape[0]
        ypred = np.zeros(n_test)
        for i in xrange(n_test):
            # L1 distances
            distances = np.sum(np.abs(self.Xtr - Xtst[i,:]), axis = 1)
            min_index = np.argmin(distances)
            ypred[i] = self.ytr[min_index]
       
        return ypred
