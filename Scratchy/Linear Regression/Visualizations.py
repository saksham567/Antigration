import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Ensure cross-platform compatibility for file paths
data_path = "Datasets"  ##Only fill when the files are present in a sub folder present

X_Train = pd.read_csv(os.path.join(data_path, "Linear_X_Train.csv"))
Y_Train = pd.read_csv(os.path.join(data_path, "Linear_Y_Train.csv"))


# Convert DataFrames to NumPy arrays
X = X_Train.to_numpy().flatten()
# Using the above instead of X_Train.values
y = Y_Train.to_numpy().flatten()

# Normalization
X_mean, X_std = np.mean(X), np.std(X)
X = (X - X_mean) / X_std

# Hypothesis Function
def hypothesis(x, theta):
    return theta[0] + x * theta[1]

# Gradient and Error Calculation (vectorized)
def gradient(X, y, theta):
    m = len(X)
    #Instead of m = X.shape[0]
    y_pred = hypothesis(X, theta)
    error = y_pred - y
    grad = np.array([np.sum(error), np.sum(error * X)]) / m
    total_error = np.mean(error ** 2)
    return grad, total_error

# Gradient Descent (vectorized update)
def GradientDescent(X, y, lr=0.1, max_steps=100):
    theta = np.zeros(2)
    error_list = []
    theta_list = np.zeros((100,2))
    for i in range(max_steps):
        grad, e = gradient(X, y, theta)
        theta -= lr * grad  # Vectorized update
        theta_list[i,0], theta_list[i,1] = theta[0], theta[1]
        error_list.append(e)
    return theta, error_list, theta_list

theta, error_list, theta_list = GradientDescent(X, y)

#Visualizations one at a time

### Visualization No. 1 ==> Tracing Loss functions
T0 = np.arange(-40,40,1)
T1 = np.arange(40,120,1)
T0, T1 = np.meshgrid(T0,T1)

J = np.zeros(T0.shape)

for i in range(J.shape[0]):
    for j in range(J.shape[1]):
        y_ = T1[i,j]*X + T0[i,j]  # y_ will retain the shape of X, as it is derived from X by only multiplying it with constants. Which is also the shape of y
        J[i,j] = np.sum((y_ - y)**2)/y.shape[0]

# fig = plt.figure()
# axes = fig.add_subplot(111,projection = '3d')
# # axes.plot_surface(T0,T1,J,cmap = 'rainbow')
# axes.contour(T0,T1,J,cmap = 'rainbow')
# plt.xlabel('Theta0')
# plt.ylabel('Theta1')
# plt.title('Path followed by the Loss function wrt changing Theta0 and Theta1')
# plt.show()

### Visualization No. 2 ==> Tracing Theta trajectory with error as elevation
# plt.plot(theta_list[:,0], color = 'orange', label = 'Theta0')
# plt.plot(theta_list[:,1], color = 'blue', label = 'Theta1')
# plt.legend()
# plt.show()

### Visualization No. 3 ==> Tracing Theta trajectory in Loss function

# fig = plt.figure()
# axes = fig.add_subplot(111,projection = '3d')
# axes.contour(T0,T1,J,cmap = 'rainbow')
# axes.scatter(theta_list[:,0],theta_list[:,1], error_list)
# plt.show()

### Visualization No. 3.1 ==> Tracing Theta trajectory in Loss function in a 2D Contour -> doesn't need to define axes
plt.contour(T0,T1,J, cmap = 'rainbow') ##Here although contour is 2D but J is required to get a feel of diminishing Loss
plt.scatter(theta_list[:,0],theta_list[:,1])
plt.show()