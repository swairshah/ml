import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.datasets import make_regression
from sklearn import datasets

def sampled_regression(sampling, X, y, I, r):
    """
    arguments :
        sampling: 'uniform' or 'importance'
        X, y
        I: Iterations
        r: row sample size
    call example:
        sampled_regression('importance',X, y, I, r)
    """
    
    m = len(X)
    sample_errs = []
    min_err = float('inf')
    best_model = None
    w = np.copy(np.ones(len(X)))
    w = w/sum(w) 
    for i in range(I):
        if sampling.lower() == 'importance':
            idx = np.random.choice(range(len(X)), r, p = w)
            w1 = w[idx]
            X1 = X[idx,:]/(m*w1[:,None])**0.5
            y1 = y[idx]/(m*w1)**0.5
        else: # uniform
            idx = np.random.randint(len(X),size=r)
            X1 = X[idx,:]
            y1 = y[idx]

        lm = LinearRegression()
        lm.fit(X1, y1)
        e = (y - lm.predict(X))**2
        error = np.mean(e)
        sample_errs.append(error)

        if error < min_err:
            min_err = error
            w = e/np.sum(e)

    return sample_errs

if __name__ == "__main__":
    m = 10000
    n = 5
    X, y, coef = make_regression(n_samples = m, n_features = n, noise = 10, coef = True)

    lm = LinearRegression()
    lm.fit(X,y)
    base_err = np.mean((lm.predict(X) - y)**2)
    print "base err %.2f" % base_err
    imp_errs = []
    uni_errs = []
    for i in range(10,100):
        imp = sampled_regression('importance', X, y, 10, 50)
        uni = sampled_regression('uniform', X, y, 10, 50)
        imp_errs.append(np.mean(imp))
        uni_errs.append(np.mean(uni))

    plt.plot(range(10,100), imp_errs, label="imp")
    plt.plot(range(10,100), uni_errs, label="uni")
    plt.legend()
    plt.show()
    #print "importance err %.2f, var %.2f" % (np.mean(imp), np.mean((imp - base_err)**2))
    #print "uniform err %.2f, var %.2f" % (np.mean(uni), np.mean((uni - base_err)**2))

    #plt.subplot(211)
    #plt.hist(uni, color="blue")
    #plt.subplot(212)
    #plt.hist(imp, color="orange")
    #plt.show()

