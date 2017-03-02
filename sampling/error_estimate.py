# Sampling uniformly to calculate error wrt optimal model 
# vs sampling with Importance sampling to calculate error wrt 
# optimal model. Does one give a 'better' estimate than the other?

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn import datasets
from sklearn.datasets import make_regression

m = 1000
n = 50
I = 100
r = 10

X, y, coef = make_regression(n_samples = m, n_features = n, noise = 10, coef = True)

lm = LinearRegression()
lm.fit(X,y)

y_pred = lm.predict(X)
e = (y - y_pred)**2
w = e/np.sum(e)
print w
print 
err_base = np.mean(e)


# uniform sampling
unif_errors = []
for i in range(I):
    idx = np.random.randint(0,len(X),size=r)
    X1 = X[idx,:]
    y1 = y[idx]
    err = np.mean((lm.predict(X1) - y1)**2)
    unif_errors.append(err)

# importance sampling
imp_errors = []
for i in range(I):
    idx = np.random.choice(range(len(X)), r, p = w)
    w1 = w[idx]
    #X1 = X[idx,:]/w1[:,None]
    #y1 = y[idx]/w1
    X1 = X[idx,:]
    y1 = y[idx]
    e = (lm.predict(X1) - y1)**2
    err = np.mean(e)
    imp_errors.append(err)


print "base error: %.2f" % err_base
print "variance importance %.2f" % np.var((imp_errors)
print "variance uniform %.2f" % np.var(unif_errors)

plt.subplot(211)
plt.hist(unif_errors, color="blue")
plt.subplot(212)
plt.hist(imp_errors, color="orange")
#plt.show()

