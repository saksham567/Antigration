import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression

X,y = make_regression(n_samples = 400, n_features = 1, n_informative = 1, noise = 1.8, random_state = 11)
y = y.reshape((-1,1))

u = np.mean(X)
std = np.std(X)
X = (X-u)/std

ones = np.ones((X.shape[0],1))
X_ = np.hstack((ones,X))

def predict(X,theta):
    return np.dot(X,theta)

def getThetaClosedForm(X,Y):
    # Y = np.mat(Y)  ##This was removed from Numpy
    Y = np.asmatrix(Y)
    firstPart = np.dot(X.T,X)
    secondPart = np.dot(X.T,Y)
    theta = np.linalg.pinv(firstPart)*(secondPart)
    return theta

theta = getThetaClosedForm(X_,y)
# print(theta)

plt.scatter(X,y)
plt.plot(X,predict(X_,theta), color = 'red', label = 'Predictions')
plt.title('Normalized Data')
plt.legend()
plt.show()