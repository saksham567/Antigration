## Using the Boston housing Data set and Linear Regression approach from Scratch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
## The boston housing dataset has been removed from Scikit learn as it presented racial self-segregation had a positive impact on house prices and it deviated from its actual goal of
## giving adequate demonstration of the validity of this assumption. Alternatively using California dataset
from sklearn.datasets import fetch_california_housing

##Fetching the dataset
california = fetch_california_housing()
##Converting them to get data and its target in an numpy array format
X = california.data
y = california.target
columns = california.feature_names

## EDA
# print(california.DESCR)
# print(columns)
# print(X.shape)
# df = pd.DataFrame(X, columns = columns)     ### DF. columns = boston.feature_names
# print(df.head())
# print(df.describe())


## Preprocessing to change the shape (add and Xo) and normalize the columns of X (convert to between 0 and 1)
u = np.mean(X, axis = 0) #Output shape will be (8,) i.e. for all the features
std = np.std(X,axis = 0) #Output shape will be (8,) i.e. for all the features
X = (X-u)/std  # This is getting normalized but that doesn't mean all the features will be between 0 and 1

mini = np.min(X, axis = 0)
maxi = np.max(X, axis = 0)
X = (X - mini)/(maxi-mini)

X = np.hstack((np.ones((20640,1)),X))


##Implementing using Vectorization
def hypothesis(X,theta):
    return np.dot(X, theta)

##Error gives values of 
def error(X,y,theta):
    y_ = hypothesis(X,theta)
    return np.mean((y-y_)**2)

def gradient(X,y, theta):
    m = X.shape[0]
    y_ = hypothesis(X, theta)
    return np.dot(X.T,(y_-y))/m  ##Imagine through first element of the output array

def gradient_descent(X,y, lr = 0.2, max_size = 1000):
    theta = np.zeros((X.shape[1],))
    error_list = []
    for _ in range(max_size):
        err = error(X,y,theta)
        error_list.append(err)
        grad = gradient(X,y,theta)
        theta = theta - lr*grad
    return theta, error_list

def batch_gradient_descent(X,y, lr = 0.2, iters =  300, batch_size = 1024):
    data = np.hstack((X, y.reshape(-1,1)))  ##reshape converts the 1 Dimensional array into 2 Dimensional array

    theta = np.zeros((X.shape[1],))
    error_list = []

    for _ in range(iters):
        
        np.random.shuffle(data)
        total_batches = data.shape[0]//batch_size

        for batch in range(total_batches):
            batch_data = data[batch*batch_size:(batch+1)*batch_size,:]

            X_batch = batch_data[:,:-1]
            y_batch = batch_data[:,-1]
    
            err = error(X_batch,y_batch,theta)
            error_list.append(err)
            grad = gradient(X_batch,y_batch,theta)
            theta = theta - lr*grad
    
    return theta, error_list


theta, error_list = gradient_descent(X,y)
theta_, error_list_ = batch_gradient_descent(X,y)

y_pred = hypothesis(X,theta)
y_pred_ = hypothesis(X,theta_)

def r2_score(y,y_pred):
    num = np.sum((y - y_pred)**2)
    demon = np.sum((y-y.mean())**2)
    score = (1 - num/demon)
    
    return score*100

print(f"The Gradient Descent R2 score is: {r2_score(y,y_pred)}")  ##Bad score maybe due to Linear Hypothesis funciton used
print(f"The Batch Gradient Descent R2 score is: {r2_score(y,y_pred_)}")

plt.plot(np.arange(len(error_list)),error_list)

plt.plot(np.arange(len(error_list_)),error_list_)
plt.show()