# Sampling uniformly to calculate error wrt optimal model 
# vs sampling with Importance sampling to calculate error wrt 
# optimal model. Does one give a 'better' estimate than the other?

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn import datasets
from sklearn.datasets import make_regression

np.random.seed(0)
m = 10000
n = 50
I = 100
r = 10
weighted = True

X, y, coef = make_regression(n_samples = m, n_features = n, noise = 10, coef = True)

lm = LinearRegression()
lm.fit(X,y)

y_pred = lm.predict(X)
e = (y - y_pred)**2
w = e/np.sum(e)
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
    if weighted:
        X1 = X[idx,:]/(m*w1[:,None])**0.5
        y1 = y[idx]/(m*w1)**0.5
    else:
        X1 = X[idx,:]
        y1 = y[idx]
    e = np.mean((lm.predict(X1) - y1)**2)

    # The following should be equlivalent to 
    # dividing x and y by sqrt(m*w1), since we are 
    # dividing the final error by m*w1.
    # doesn't give the same answer.
    
    #X1 = X[idx,:]
    #y1 = y[idx]
    #e = np.mean((lm.predict(X1) - y1)**2)
    #if weighted:
    #    e = e/(m*w1)
    imp_errors.append(e)

print "base error: %.2f" % err_base
print "imp err", np.mean(imp_errors), "uni err", np.mean(unif_errors)
#print "variance importance %.2f" % np.var(imp_errors)
#print "variance uniform %.2f" % np.var(unif_errors)
print "variance importance %.2f" % np.mean((imp_errors - err_base)**2)
print "variance uniform %.2f" % np.mean((unif_errors - err_base)**2)

#plt.subplot(211)
#plt.hist(unif_errors, color="blue")
#plt.subplot(212)
#plt.hist(imp_errors, color="orange")
#plt.show()

