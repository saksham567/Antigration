## Locally weighted Linear Regression
import numpy as np
import pandas as pd
# from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
plt.ion()  ###Interactive plots

### make_regression won't work here as we want to capture non-linear trends in noisy data, so below can be used to synthesize such a dataset
np.random.seed(42)
X = np.linspace(0, 10, 100)
y_true = np.sin(X)
noise = np.random.normal(scale=0.25, size=X.shape) ##Produces random numbers from a gaussian or normal distribution
y_noisy = y_true + noise

## Normalized data
X = (X-np.mean(X))/np.std(X)

# ## Plot the noisy data
# plt.scatter(X, y_noisy, label="Noisy Data", alpha=0.7)
# plt.plot(X, y_true, color='black', label="True Function")
# plt.legend()
# plt.title("Synthetic Data for LOWESS")
# plt.show()


## Getting the weights
def getW(query_point, X, tau):
    M = X.shape[0]
    W = np.asmatrix(np.eye(M))
    x = query_point

    for i in range(M):
        xi = X[i]
        # X[i] itself will be a vector and to take its squar we will have to do: 
        W[i,i] = np.exp(np.dot((xi - x), (xi - x).T)/(-2*tau*tau))

    return W

X = np.asmatrix(X).reshape((-1,1))
y_noisy = np.asmatrix(y_noisy).reshape((-1,1))

def predict(X, y, query_x, tau):
    ##Preparing X matrix
    M = X.shape[0]
    ones = np.ones((M,1))
    X_ = np.hstack((ones, X))

    ###Preparing qx matrix
    qx = np.asmatrix([1, query_x])

    ##Getting the whole weight matrix for that single query point
    W = getW(qx, X_, tau)  
    ## Larger tau means higher bandwidth -> far off points will also start to have the influence

    ###Calculating theta for the specific query point using closed form solution
    theta = np.linalg.pinv((X_.T*(W*X_)))*(X_.T*(W*y))
    ### Return the prediction for that specific query point
    pred = np.dot(qx, theta)

    return theta, pred

def plotPrediction(tau):
    X_test = np.linspace(-1.5,1.5, 30)
    y_test = []

    for qx in X_test:
        theta, pred = predict(X,y_noisy,qx, tau)
        y_test.append(pred[0][0])
    
    y_test = np.array(y_test)

    XO = np.array(X)
    yO = np.array(y_noisy)

    plt.scatter(XO,yO)
    plt.scatter(X_test, y_test, color = 'red')
    plt.draw()
    plt.pause(0.5)
    plt.clf()


tau = [0.2,0.5,1,3,5,10]

for t in tau:
    plotPrediction(t)
    