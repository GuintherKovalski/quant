import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from indicators import features
import numpy as np

DEQUE_LEN = 1
FRACAO = 1000
WINDOW = 22

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

BITCOIN = pd.read_csv('../data/GSPC.csv')


def series(data, t, n):
    d = t - n + 1
    block1 = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    return block1

def buy_logic(der,buy_order,B1,B2):
    return sum(der) + B2*sum(der)*sum(der) > B1  and len(buy_order)<DEQUE_LEN 

def sell_logic(der,buy_order,S1,S2):
    return sum(der)+ S2*sum(der)*sum(der) < S1  and len(buy_order)>0

parameters = [1,0.1,-0.1,0.0001,0.00001]
FRACAO = 1000
DEQUE_LEN = 1
D1,B1,S1,B2,S2 = parameters
D1 = int(D1)
history,sell,buy,hold  = [],[],[],[]
profits,buy_order,los_mean,win_mean,op = [],[],[],[],[]
win = 0
lose = 0
window = 300
horizon = 30

def show(xi,xf):
    plt.figure(num = 0, figsize=(5, 3), dpi=200)
    plt.plot(DATA, linewidth=0.2) 
    plt.scatter(sell,np.array(DATA)[sell], color = 'green', s = 2)
    plt.scatter(buy,np.array(DATA)[buy], color = 'red',s=2)
    plt.plot(MEAN)
    #lt.scatter(sell, np.array(DATA)[sell], s=abs(np.array(history))*100, alpha=0.5) 
    plt.xlim(xi,xf)
    yi,yf= min(DATA[xi:xf]),max(DATA[xi:xf])
    #plt.plot(sell,np.array(cumulative)[1:]*(yf-yi)+yi) 
    plt.ylim(yi,yf)
    plt.legend(['price','MA','sell','buy'],loc='best')

#show(110000,110400)
#show(0,600)
#show(299200,299240)


#quanto tempo leva entre uma compra e outra?
'''
mean_time = []
buy.insert(0,0)
for i in range(len(buy)-1):
    mean_time.append(buy[i+1]-buy[i])
mean_time = np.array(mean_time)
mean_time.mean()
mean_time.std()      
plt.plot(mean_time)
'''
############################################

import json
import requests
import time
import sys
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.pyplot as plt
import requests
import time
from datetime import datetime

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

import ETS

def balance():
    session = requests.session()
    session.auth = ("cUH6YEYiZ0DkZHdMbu9bvCmyyxHi9ErS", "AHAhMcMwI41YbDar69likScAJXSYlRHO")
    saldo = session.get('https://api.hitbtc.com/api/2/trading/balance').json()
    for i in range(len(saldo)):
        if saldo[i]['currency'] == 'BTC':
            BTC = float(saldo[i]['available'])
        if saldo[i]['currency'] == 'USD': 
            USD = float(saldo[i]['available'])     
    print('saldo BTC:',BTC)
    print('saldo USD:',USD)

def monitor(key):
    url = "https://api.hitbtc.com/api/2/public/ticker/BTCUSD"
    r = requests.get(url)
    retorno = {}
    resposta = r.json()
    if key == 'last':
        last = float(resposta[key])
    else:
        last = resposta[key]
    return last

def sell_btc(ORDER_SIZE_USD): 
    '''ORDER_SIZE_BTC = quantos BTC's serão vendidos'''
    price = monitor('last')
    simbolo = 'BTCUSD'
    session = requests.session()
    session.auth = ("cUH6YEYiZ0DkZHdMbu9bvCmyyxHi9ErS", "AHAhMcMwI41YbDar69likScAJXSYlRHO")
    orderData = {'symbol':simbolo, 
                 'side': 'sell', 
                 'quantity': str(ORDER_SIZE_USD/price), 
                 'price': str(ORDER_SIZE_USD) }
    #r = session.post('https://api.hitbtc.com/api/2/order', data = orderData)        
    return price

def buy_btc(ORDER_SIZE_USD):
    price = (90*float(monitor('last'))+5*float(monitor('high'))+5*float(monitor('low')))/100
    simbolo = 'BTCUSD'
    session = requests.session()
    session.auth = ("cUH6YEYiZ0DkZHdMbu9bvCmyyxHi9ErS", "AHAhMcMwI41YbDar69likScAJXSYlRHO")
    #RESTO = np.ceil(ORDER_SIZE_USD/price*1e6 % 10)
    #CENTAVOS = RESTO/1000000*price
    #USD = ORDER_SIZE_USD+CENTAVOS
    orderData = {'symbol':simbolo,
                 'side': 'buy',
                 'quantity': str(ORDER_SIZE_USD/price),
                 'price': str(price)}
    print('order placed:')
    print('BTC:',ORDER_SIZE_USD/price)
    print('USD:',ORDER_SIZE_USD)
    print('COTAÇÃO:',price)

    #r = session.post('https://api.hitbtc.com/api/2/order', data = orderData)        
    return price


def arima_pred(lista,parameters,INIT=195,FIM=231): 
    p,d,q,P,D,Q,s,a1,a2,a3,a4 = parameters
    train = np.array(lista) 
    param = (int(p),int(d),int(q))
    param_seasonal = (int(P),int(D),int(Q),int(s))
    TREND = [int(a1),int(a2),int(a3),int(a4)]
    train =np.reshape(np.array(lista).astype(float),-1,1)
    mod = sm.tsa.statespace.SARIMAX(train,order=param,seasonal_order=param_seasonal, enforce_stationarity=True,enforce_invertibility=True,initialization='approximate_diffuse')
    results = mod.fit()
    pred0 = results.get_prediction(start= INIT,end = FIM-1, dynamic=False)
    pred0_ci = pred0.conf_int()
    y_pred = pred0.predicted_mean
    x = np.array(range(INIT,FIM))
    return x,y_pred,pred0_ci
    
def plot(lista,x_pred,y_pred,ci_i,ci_s,dydx,SELL,BUY):
    file_name = str(time.time()).split('.')[0]
    p = plt.figure(figsize=(6,4))
    plt.style.use('classic') 
    plt.plot(x_pred,y_pred)
    plt.fill_between(x_pred,ci_i, ci_s,alpha=0.5, color='b')
    plt.plot(lista,linewidth=2)
    plt.ylabel('USD/BTC')
    plt.xlabel('dydx='+str(dydx))
    plt.xlim(0,x_pred[-1])
    plt.grid()
    if SELL:
        plt.savefig('img/'+str(file_name)+'SELL_arima.png')
    elif BUY:
        plt.savefig('img/'+str(file_name)+'BUY_arima.png')
    plt.show()
    plt.clf()
    plt.close(p)    

def candles(window,period):
    period = str(period)
    window = WINDOW[0]
    url = 'https://api.hitbtc.com/api/2/public/candles/BTCUSD?period=' + window+'&limit='+period
    r = requests.get(url)
    retorno = {}
    resposta = r.json()
    MEAN,TIMESTAMP = [ ],[ ]
    for i in range(len(np.array(resposta))):
        MEAN.append((float(np.array(resposta)[i]['max'])+float(np.array(resposta)[i]['min']))/2)
        TIMESTAMP.append(np.array(resposta)[i]['timestamp'])
    return list(MEAN), TIMESTAMP


def GPR(lista,END):  
    y = np.array(lista)
    X = np.linspace(1, y.shape[0] , num=y.shape[0] ).reshape(-1, 1)
    # Kernel with optimized parameters
    k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
    k2 = 2.0**2 * RBF(length_scale=100.0) \
    * ExpSineSquared(length_scale=1.0, periodicity=100,
                     periodicity_bounds="fixed")  # seasonal component
    # medium term irregularities
    k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
    k4 = 0.1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.1**2,
                  noise_level_bounds=(1e-3, np.inf))  # noise terms
    kernel = k1 + k2 + k3 + k4
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                              normalize_y=True)
    gp.fit(X, y)   
    X_ = np.linspace(X.min(), END, END)[:, np.newaxis]
    y_pred, y_std = gp.predict(X_, return_std=True)
    X_ =  X_.reshape(-1,)
    return X_,y_pred, y_std

#x_gpr,gpr_y_pred, y_std, = GPR(lista,660)
#plot(lista,x_gpr,gpr_y_pred,gpr_y_pred+y_std,gpr_y_pred-y_std) 

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    

''' SEGUNDO A SEGUNDO '''

WINDOW = ['M1','M5','M15','M30','H1','H4','D1','W1','M1']
horizon = 30
window = 300
timeframe = WINDOW[0]

BTC = np.array(BITCOIN.high)*0.5 + np.array(BITCOIN.low)*0.5

sell = []
when = []
buy = []
trade_history = []
history = []
lista = []
deriv_hist =[]
save = time.time()
BUY = False
SELL = False


for i in range(100000,101002):
    lista  = BTC[i*WINDOW:i*WINDOW+WINDOW]
    last = lista[-1]
    #################### INDICATORS ########################
    if i ==0:
        print('Runing ARIMA')
        parameters = 1,1,1,1,2,1,horizon,1,1,1,1
        x_pred,y_pred,pred_ci = arima_pred(lista,parameters,INIT=len(lista),FIM=len(lista)+horizon)
        
    #print('Runing Gaussian Process')
    #x_gpr,gpr_y_pred, y_std, = GPR(lista,660)
    
    #ETS_ypred = ETS.exponential_smooth(lista, 601, 660)

    der = np.array(lista)
    der = list(der[1:] - lista[:len(der)-1])
    der = np.array(der)
    
    ######################## BUY ###########################
    
    if len(buy)<1:
        dydx = (y_pred[-horizon:].mean()-lista[-1])/(horizon/2)
        if (dydx>1.1) or (der[-1]<-20):
            buy.append(lista[-1])
            trade_history.append((buy[-1],str(datetime.fromtimestamp(time.time())),'buy'))
            print('comprado por:',buy)
            BUY = True
        
    ######################## SELL ##########################   

    elif len(buy)>0:    
        if last-buy[0]>(1.01*buy[0]+last*(0.07/100)) or last<0.995*buy[0]:
            if lista[-1]+5< lista[-2]:
                sell.append(lista[-1])
                trade_history.append((sell[-1],str(datetime.fromtimestamp(time.time())),'sell'))
                history.append((sell[-1]-buy[0])*(1-0.07/100))
                when.append(str(datetime.fromtimestamp(time.time())))
                buy.pop(0)
                sell_btc(10)
                SELL = True
   
            
    ######################## SHOW ##########################
    #plot(lista,x_pred,y_pred,pred_ci[:,0],pred_ci[:,1],dydx,SELL,BUY)
    #plot(lista,x_gpr,gpr_y_pred,gpr_y_pred+y_std,gpr_y_pred-y_std) 
    #plot(lista,x_pred,ETS_ypred,ETS_ypred,ETS_ypred)
    print('arima derivate:',dydx)
    print('Preço atual:',lista[-1])
    print('Comprado por:',buy)
    print('Total:',sum(history))
    deriv_hist.append((y_pred[-horizon:].mean()-lista[-1],monitor('timestamp')))    
                                
HFT(horizon,window,timeframe)











