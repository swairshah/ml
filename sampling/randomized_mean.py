import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn import datasets
from sklearn.datasets import make_regression
from numpy.linalg import norm


def experiment():
    m = 100
    n = 10
    r = 100
    x = np.random.uniform(0, 1000, size = m*n)
    X = np.reshape(x, (m,n))
    normX = np.linalg.norm(X)
    prob = np.linalg.norm(X, axis = 1)**2/normX**2
    mean = np.mean(X, axis = 0)
    
    X_ = X/(m*prob[:,None])
    uni_means = []
    imp_means = []
    for i in range(100):
        idx1 = np.random.randint(0, m, size = r)
        m1 = np.mean(X[idx1,:], axis=0)
        uni_means.append(norm(m1))
    
        idx2 = np.random.choice(range(m), size = r, p=prob/sum(prob) )
        m2 = np.mean(X_[idx2,:],axis = 0)
        imp_means.append(norm(m2))
        #print norm(mean), norm(m1), norm(m2)

    plt.subplot(211)
    plt.hist(uni_means, color="blue", label = str(np.var(uni_means)))
    plt.legend()
    plt.subplot(212)
    plt.hist(imp_means, color="orange", label = str(np.var(imp_means)))
    plt.legend()
    plt.show()
    #return (np.var(uni_means) - np.var(imp_means))
"""
better = []
for i in range(100):
    ret = experiment()
    better.append(ret)

better = np.array(better)
print len(better[better > 0])
print better
"""
print experiment()
