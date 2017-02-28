import numpy as np
from cs231nlib.utils import *
from nn2 import NearestNeighbor

Xtr, Ytr, Xte, Yte = load_CIFAR10("datasets/cifar-10-batches-py/")
# flatten out images
Xtr_flat = Xtr.reshape(Xtr.shape[0], 32*32*3)
Xte_flat = Xte.reshape(Xte.shape[0], 32*32*3)
nn = NearestNeighbor()
nn.train(Xtr_flat, Ytr)
Yte_predict = nn.predict(Xte_flat)

print 'accuracy: %f' % ( np.mean(Yte_predict == Ytr) )
