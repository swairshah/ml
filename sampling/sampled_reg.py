import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn import datasets

def regression(X, y):
    pseudoinv = np.linalg.pinv(np.dot(X.T,X))
    a = np.dot(np.dot(pseudoinv,X.T),y)
    return a

def reg(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    #line = slope*x+intercept
    return np.array(slope, intercept)


def sampled_regression(sampling, X, y, I, r):
    """
    arguments :
        sampling: 'uniform' or 'importance'
        X, y
        I: Iterations
        r: row sample size
        
        call example:
        sampled_regression(sampling='importance',X=X, y=y, I=10, r=10)
    """

    sample_errs = []
    min_err = float('inf')
    best_model = None
    w = np.copy(np.ones(len(X)))
    w = w/sum(w)
    
    for i in range(I):
        if sampling.lower() == 'importance':
            idx = np.random.choice(range(len(X)), r, p = w)
        else: # uniform
            idx = np.random.randint(len(X),size=r)

        X1 = X[idx,:]/w[idx]
        y1 = y[idx]/w[idx]
        a1 = regression(X1,y1)

        e = (y - np.dot(X, a1))**2
        error = np.mean(e)
        sample_errs.append(error)

        if error < min_err:
            best_model = a1
            min_err = error
            w = e/np.sum(e)

    return sample_errs

if __name__ == "__main__":
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    lm = LinearRegression()
    lm.fit(X,y)
    base_err = np.mean((lm.predict(X) - y)**2)
    print "base err %.2f" % base_err

    imp = sampled_regression('importance', X, y, 10, 10)
    uni = sampled_regression('uniform', X, y, 10, 10)
    #print imp
    #print uni
    plt.plot(imp)
    plt.plot(uni, c = 'orange')
    plt.legend(['importance','uniform'])
    plt.show()
