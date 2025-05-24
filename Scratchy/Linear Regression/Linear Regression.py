import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure cross-platform compatibility for file paths
data_path = ""  ##Only fill when the files are present in a sub folder present

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
    for _ in range(max_steps):
        grad, e = gradient(X, y, theta)
        theta -= lr * grad  # Vectorized update
        error_list.append(e)
    return theta, error_list

# Execute Gradient Descent
theta, error_list = GradientDescent(X, y)
# print(theta)

###Test data prepare
x_test = pd.read_csv(os.path.join(data_path, "Linear_X_Test.csv")).to_numpy().flatten()
y_pred = hypothesis(x_test, theta)
# ##Saving to a file
df = pd.DataFrame(data = y_pred, columns = ["y"])
df.to_csv(os.path.join(data_path, "Linear_y_Test.csv"),index = False)

##Plotting everything
plt.scatter(X,y)
plt.plot(X,hypothesis(X,theta), color = 'orange', label = 'Training_predictions')
plt.plot(x_test,y_pred, color = 'green', label = 'Test_predictions')
plt.legend()
plt.show()

# Plot Error Reduction Over Iterations
# plt.plot(error_list)
# plt.xlabel("Iterations")
# plt.ylabel("Error")
# plt.title("Error Reduction Over Iterations")
# plt.show()