import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import r2_score 
from matplotlib.pyplot import figure
import seaborn as sns
from scipy.optimize import differential_evolution


def metrics(testY,y_pred): 
    testY=testY.reshape(-1,1)
    y_pred=y_pred.reshape(-1,1)
    resid = np.subtract(testY, y_pred)
    RMSE = abs(resid).mean()
    return RMSE

def cost(parameters, *args):  
    parameters = 0.1, 0.1,0.0001,30 
    #parameters = a
    alpha,beta,gamma, periodo = parameters
    dataset = args
    #d = data
    dataset = np.reshape(np.array(dataset).astype(float),-1,1)
    Cost = [] 
    train=np.reshape(np.array(dataset).astype(float),-1,1)
    ets_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods= int( periodo))
    ets_fit = ets_model.fit(smoothing_level= alpha, smoothing_slope= beta, smoothing_seasonal= gamma)         
    y_pred = ets_fit.predict(start=1, end=len(dataset))
    RMSE = metrics(np.array(dataset),y_pred)    
    return RMSE*100

#producing "experimental" data 
def optimize(data):
    #             alpha     beta        gamma    periodo
    bounds = [(0.0001, 0.9), (0.0001, 0.5), (0, 0.6),(3, 120)]
    args = data
    result = differential_evolution(cost, bounds, args=args,strategy='rand2exp',maxiter = 10, popsize =20  ,recombination = 0.5)
    return result.x



def exponential_smooth(lista, INIT, END):
    parameters  = optimize(lista)
    alpha,beta,gamma, periodo = parameters    
    Cost = [] 
    train=np.reshape(np.array(lista).astype(float),-1,1)
    ets_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods= int( periodo))
    ets_fit = ets_model.fit(smoothing_level= alpha, smoothing_slope= beta, smoothing_seasonal= gamma)         
    y_pred = ets_fit.predict(start=INIT, end=END)
    return y_pred



    
  


  