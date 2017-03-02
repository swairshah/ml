import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn import datasets
from sklearn.datasets import make_regression


def experiment():
    m = 1000
    n = 10
    r = 100
    x = np.random.uniform(0, 1000, size = m*n)
    X = np.reshape(x, (m,n))
    prob = x/sum(x)
    print x/prob
    mean = np.mean(x)

    uni_means = []
    imp_means = []
    print x_
    for i in range(100):
        idx1 = np.random.randint(0, m, size = r)
        m1 = np.mean(x[idx1])
        uni_means.append(m1)
    
        idx2 = np.random.choice(range(len(x)), size = r, p=prob )
        m2 = np.mean(x_[idx2])
        imp_means.append(m2)
        print mean, m1, m2

    plt.subplot(211)
    plt.hist(uni_means, color="blue", label = str(np.var(uni_means)))
    plt.legend()
    plt.subplot(212)
    plt.hist(imp_means, color="orange", label = str(np.var(imp_means)))
    plt.legend()
    #plt.show()
    return (np.var(uni_means) - np.var(imp_means))

"""
better = []
for i in range(100):
    ret = experiment()
    better.append(ret)

better = np.array(better)
print len(better[better > 0])
print better
"""
experiment()
