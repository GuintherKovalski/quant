import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas
import math
from sklearn.svm import SVR
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score 

def cost(parameters, *arg):
    dataset = arg
    dataset = np.reshape(np.array(dataset).astype(float),-1,1)
    K_index,  C,  coef0,degree,tol,epsilon,gamma,look_back = parameters
    look_back = int(look_back)
    kernel = ['linear', 'poly', 'rbf', 'sigmoid'] #0 a 4
    Kernel = kernel[int(K_index)]  
    Cost= []
    trainX, trainY = create_dataset(dataset, look_back)
    trainX = trainX.reshape(trainX.shape[0],look_back)
    trainY = trainY.reshape(-1, 1) 
    svr = SVR(kernel= Kernel, C=int(C), coef0 = coef0, 
              degree=int(degree),tol=tol, gamma=gamma,
              max_iter = 200000,epsilon= 0.01,cache_size = 5)       
    model = svr.fit(trainX,trainY.ravel())      
    y_hat = model.predict(trainX)
    rmse = metrics(trainY,y_hat) 
   
    return rmse

def optimize(Data):
    #initial guess for variation of parameters                 
    #        |Kernel_index|     C,     |   coef0   |   degree  |   tol      |   epsilon  |    gamma   |  look_back            
    bounds = [  (1, 3.9),   (1e4,1e10),    (0,3),     (1,6),  (0.0001, 0.001),(0.01,0.5),  (0.05,2),  (6,90)]

    args = np.array(Data) #  trainX, trainY, TestX, TestY
    parameters = differential_evolution(cost, bounds, args=args,strategy='rand2exp',maxiter = 10, popsize = 10  ,recombination = 0.20)
    return parameters.x

def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		dataX.append(dataset[i:(i+look_back)])
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

def metrics(testY,y_pred): 
    testY=testY.reshape(-1,1)
    y_pred=y_pred.reshape(-1,1)
    resid = np.subtract(testY, y_pred)
    RMSE = abs(resid).mean()
    return RMSE

lista,timestamp=candles(WINDOW[4],600)
x = np.array(lista)
MAX = ((x - x.min()).max()) 
MIN = x.min()
y = (x - x.min())/((x - x.min()).max()) 
#x = y*MAX+MIN

param = optimize(y)

parameters = 3.9,1e10,0.001,3,6,0.5,2,6.9



