import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X_Train = pd.read_csv("Datasets\Linear_X_Train.csv")
Y_Train = pd.read_csv("Datasets\Linear_Y_Train.csv")

X = X_Train.values
Y = Y_Train.values

theta_list = np.load("Datasets\ThetaList.npy")
T0 = theta_list[:,0]
T1 = theta_list[:,1]

##For interactive plots use plt.ion() function
plt.ion()
for i in range(0,50,2): #You need to cover it in a loop for being interactive, although theta_list has line path for 100 iterations, we limit to 50    
    y_ = T1[i]*X + T0[i]
    plt.scatter(X,Y)
    plt.plot(X,y_,'red')
    plt.draw()
    plt.pause(1)
    plt.clf()