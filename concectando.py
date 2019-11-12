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
#plt.switch_backend('agg')
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


def trade_history():
    session = requests.session()
    session.auth = ("cUH6YEYiZ0DkZHdMbu9bvCmyyxHi9ErS", "AHAhMcMwI41YbDar69likScAJXSYlRHO")
    history = session.get('https://api.hitbtc.com/api/2/history/trades?symbol=BTCUSD').json()
    op =    []
    qtd =   []
    price = []
    fee =   []
    Time =  []
    usd =   []
    for i in range(len(history)):    
        op.append(     history[i]['side'])
        qtd.append(    float(history[i]['quantity']))
        price.append(  float(history[i]['price']))
        fee.append(    float(history[i]['fee']))
        Time.append(   history[i]['timestamp'])
        usd.append(    float(history[i]['quantity'])*float(history[i]['price']))
    hist = pd.DataFrame({'time':Time,'op':op,'price':price,'usd':usd,'btc':qtd,'taxa (USD)':fee})#.to_csv('historico.csv')
    for i in range(len(hist.time)):
        hist['time'].iloc[i] = hist['time'].iloc[i].split('T')[0] 
    hist.to_csv('historico.csv')
    hist.to_excel('history.xlsx')
    #today = str(datetime.today()).split(' ')[0]
    #mask = (pd.to_datetime(hist['time']) >= today)
    #df = df[mask]
  
  
          
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



def sell_btc(BTC_quantity): 
    '''ORDER_SIZE_BTC = quantos BTC's serão vendidos'''
    price = monitor('last')
    simbolo = 'BTCUSD'
    session = requests.session()
    session.auth = ("cUH6YEYiZ0DkZHdMbu9bvCmyyxHi9ErS", "AHAhMcMwI41YbDar69likScAJXSYlRHO")
    orderData = {'symbol':simbolo, 
                 'side': 'sell', 
                 'quantity': str(BTC_quantity), 
                 'price': str(price) }
    r = session.post('https://api.hitbtc.com/api/2/order', data = orderData) 
    print('order placed:')
    print('BTC:',BTC_quantity*price)
    print('USD:',BTC_quantity*price)
    print('COTAÇÃO:',price)       
    return price

def buy_btc(BTC_quantity):
    price = (95*float(monitor('last'))+2*float(monitor('high'))+3*float(monitor('low')))/100
    simbolo = 'BTCUSD'
    session = requests.session()
    session.auth = ("cUH6YEYiZ0DkZHdMbu9bvCmyyxHi9ErS", "AHAhMcMwI41YbDar69likScAJXSYlRHO")
    #RESTO = np.ceil(ORDER_SIZE_USD/price*1e6 % 10)
    #CENTAVOS = RESTO/1000000*price
    #USD = ORDER_SIZE_USD+CENTAVOS
    orderData = {'symbol':simbolo,
                 'side': 'buy',
                 'quantity': str(BTC_quantity),
                 'price': str(price)}
    print('order placed:')
    print('BTC:',BTC_quantity*price)
    print('USD:',BTC_quantity*price)
    print('COTAÇÃO:',price)

    r = session.post('https://api.hitbtc.com/api/2/order', data = orderData)        
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
    pred0_ci = pred0.conf_int(alpha=0.5)
    y_pred = pred0.predicted_mean
    x = np.array(range(INIT,FIM))
    return x,y_pred,pred0_ci
    
def plot(lista,x_pred,y_pred,ci_i,ci_s,dydx,SELL,BUY):
    p = plt.figure(figsize=(10,6))
    plt.style.use('classic') 
    plt.plot(x_pred,y_pred)
    plt.plot(lista,linewidth=2)
    plt.fill_between(x_pred,ci_i, ci_s,alpha=0.5, color='b')
    plt.ylabel('USD/BTC')
    plt.xlabel('dydx='+str(dydx))
    plt.xlim(0,x_pred[-1])
    plt.grid()
    
    if SELL:
        TITLE = 'SELL'
    elif BUY:
        TITLE = 'BUY'
    else:
        TITLE = ' '
    plt.title(TITLE+' '+str(datetime.now()).split('.')[0])
    file_name = str(time.time()).split('.')[0]
    plt.savefig('img/'+str(file_name)+TITLE+'.png')  
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

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
''' SEGUNDO A SEGUNDO '''

WINDOW = ['M1','M3','M5','M15','M30','H1','H4','D1','W1']
horizon = 30
window = 300
timeframe = WINDOW[0]

def HFT(horizon,window,timeframe):
    sell = []
    when = []
    buy = []
    trade_history = []
    history = []
    lista = []
    lista.append(monitor('last'))
    deriv_hist =[]
    save = time.time()
    BUY = False
    SELL = False
    while 1:
        try:
            last = (monitor('last'))   
            if len((lista))<window:
                lista,timestamp=candles(WINDOW[4],window)
                #lista.append(last)
                print('Esperando acumular'+ str(window)+ 'valores: '+str(len(lista))+'/300')
                           
            elif lista[-1] != last:
                
                #################### INDICATORS ########################
                print('Runing ARIMA')
                parameters = 1,1,1,1,2,1,horizon,1,1,1,1
                x_pred,y_pred,pred_ci = arima_pred(lista,parameters,INIT=len(lista),FIM=len(lista)+horizon)
                
                print('Runing Gaussian Process')
                #x_gpr,gpr_y_pred, y_std, = GPR(lista,660)
                
                #ETS_ypred = ETS.exponential_smooth(lista, 601, 660)
    
                der = np.array(lista)
                der = list(der[1:] - lista[:len(der)-1])
                der = np.array(der)
                
                ######################## BUY ###########################
                
                if len(buy)<1:
                    dydx = (y_pred[-horizon:].mean()-lista[-1])/(horizon/2)
                    if (dydx>1.1) or (der[-1]<-20):
                        #buy.append(buy_btc(50))
                        trade_history.append((buy[-1],str(datetime.fromtimestamp(time.time())),'buy'))
                        print('comprado por:',buy)
                        #BUY = True
                    
                ######################## SELL ##########################   
            
                elif len(buy)>0:    
                    if last-buy[0]>(100+last*(0.07/100)) or last<(buy[0]-last*(1/100)) :
                        #sell.append(sell_btc(50))
                        trade_history.append((sell[-1],str(datetime.fromtimestamp(time.time())),'sell'))
                        history.append((sell[-1]-buy[0])*(1-0.07/100))
                        when.append(str(datetime.fromtimestamp(time.time())))
                        buy.pop(0)
                        #sell_btc(10)
                        #SELL = True
                #else:
                    #BUY = False
                    #SELL = False
                          
                        
                ######################## SHOW ##########################
                plot(lista,x_pred,y_pred,pred_ci[:,0],pred_ci[:,1],dydx,SELL,BUY)
                #plot(lista,x_gpr,gpr_y_pred,gpr_y_pred+y_std,gpr_y_pred-y_std) 
                #plot(lista,x_pred,ETS_ypred,ETS_ypred,ETS_ypred)
                print('arima derivate:',dydx)
                print('Preço atual:',lista[-1])
                print('Comprado por:',buy)
                print('Total:',sum(history))
                balance()
                deriv_hist.append((y_pred[-horizon:].mean()-lista[-1],monitor('timestamp')))    
                lista.append(last)
                
              
            while len(lista)>window:
                lista.pop(0)
            if time.time()-save>180:
                save = time.time()
                df_hist = pd.read_csv('history.csv')
                df_trade_hist = pd.read_csv('trade_history.csv')
                df_hist_new = pd.DataFrame({'profit':history,'time':when})   .to_csv('history.csv')
                df_trade_hist_new = pd.DataFrame(trade_history,columns = ['price','time','action'])  .to_csv('trade_history.csv')
                pd.concat([df_hist, df_hist_new], axis=0).iloc[:,1:].to_csv('history.csv')
                pd.concat([df_trade_hist, df_trade_hist_new], axis=0).iloc[:,:].to_csv('trade_history.csv') 
                trade_history,history = [],[]
        except ConnectionError:
            print('ConnectionError')
            time.sleep(30)
            lista = []
            #lista.append(monitor('last'))
            
#HFT(horizon,window,timeframe)



BTC_quantity = 0.00360 
WINDOW = ['M1','M3','M5','M15','M30','H1','H4','D1','W1']
horizon = 30
window = 300
timeframe = WINDOW[3]

def DT(horizon,window,timeframe):
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
    last = 0
    while 1:
        try:
            lista,timestamp=candles(WINDOW[4],window)
            #x = moving_average(lista, 4)
            #lista.append(last)
            
            #last = lista[-1]         
            if lista[-1] != last:
                last = lista[-1]
                
                #################### INDICATORS ########################
                print('Runing ARIMA')
                parameters = 1,1,1,1,2,1,horizon,1,1,1,1
                x_pred,y_pred,pred_ci = arima_pred(lista,parameters,INIT=len(lista),FIM=len(lista)+horizon)
                #x_pred,y_pred,pred_ci = arima_pred(x,parameters,INIT=len(x),FIM=len(lista)+horizon)

                dydx = (y_pred[-horizon:].mean()-lista[-1])/(horizon/2)

                #print('Runing Gaussian Process')
                #x_gpr,gpr_y_pred, y_std, = GPR(lista,660)
                #ETS_ypred = ETS.exponential_smooth(lista, 601, 660)
    
                der = np.array(lista)
                der = list(der[1:] - lista[:len(der)-1])
                der = np.array(der)
                
                ######################## BUY ###########################

                if len(buy)<1:
                    if (dydx>1.1) or (der[-1]<-15):
                        buy.append(buy_btc(0.00360))
                        trade_history.append((buy[-1],str(datetime.fromtimestamp(time.time())),'buy'))
                        print('comprado por:',buy)
                        BUY = True
                    
                ######################## SELL ##########################   
            
                elif len(buy)>0:    
                    if last-buy[0]>(100+last*(0.07/100)) or last<(buy[0]-last*(0.2/100)):
                        if lista[-1]< lista[-2]:
                            sell.append(sell_btc(0.00360))
                            trade_history.append((sell[-1],str(datetime.fromtimestamp(time.time())),'sell'))
                            history.append((sell[-1]-buy[0])*(1-0.07/100))
                            when.append(str(datetime.fromtimestamp(time.time())))
                            buy.pop(0)
                            SELL = True
                else:
                    BUY = False
                    SELL = False
                          
                ######################## SHOW ##########################
                plot(lista,x_pred,y_pred,pred_ci[:,0],pred_ci[:,1],dydx,SELL,BUY)
                #plot(x,x_pred,y_pred,pred_ci[:,0],pred_ci[:,1],dydx,SELL,BUY)
                #plot(lista,x_gpr,gpr_y_pred,gpr_y_pred+y_std,gpr_y_pred-y_std) 
                #plot(lista,x_pred,ETS_ypred,ETS_ypred,ETS_ypred)
                print('arima derivate:',dydx)
                print('Preço atual:',lista[-1])
                print('Comprado por:',buy)
                print('Total:',sum(history))
                balance()
                deriv_hist.append((y_pred[-horizon:].mean()-lista[-1],monitor('timestamp')))    
                lista.append(last)      
            else:
                print('Esperando passagem do candle')
                time.sleep(5)
        except ConnectionError:
            print('ConnectionError')
            time.sleep(30)
            lista = []
            #lista.append(monitor('last'))

DT(horizon,window,timeframe)


'''
            if time.time()-save>180:
                save = time.time()
                #df_hist = pd.read_csv('history_dt.csv')
                try:
                    df_trade_hist = pd.read_csv('trade_history_dt.csv')
                    df_hist_new = pd.DataFrame({'profit':history,'time':when}).to_csv('history_dt.csv')
                    df_trade_hist_new = pd.DataFrame(trade_history,columns = ['price','time','action'])  .to_csv('trade_history.csv')
                    pd.concat([df_hist, df_hist_new], axis=0).iloc[:,1:].to_csv('history_dt.csv')
                    pd.concat([df_trade_hist, df_trade_hist_new], axis=0).iloc[:,:].to_csv('trade_history.csv') 
                    trade_history,history = [],[]
                except:
                    df_hist_new = pd.DataFrame({'profit':history,'time':when}).to_csv('history_dt.csv')
                    df_trade_hist_new = pd.DataFrame(trade_history,columns = ['price','time','action']).to_csv('trade_history.csv')
                    trade_history,history = [],[]
'''
''' MINUTO A MINUTO '''
'''
MEAN, LAST = candles(WINDOW[2])
x,y_pred = arima_pred(MEAN,INIT=1000,FIM=1040,PERIODOS = 12)
plt.plot(MEAN)
plt.plot(x,y_pred)


 
while 1:
    try:
        lista, CURRENT_TIME = candles(WINDOW[0])           
        if CURRENT_TIME != LAST_TIME:
            LAST_TIME = CURRENT_TIME
            last = lista[-1]
            minuto = str(datetime.fromtimestamp(time.time())).split(':')[1]
    
            #################### INDICATORS ########################
            parametros = 1,1,1,2,2,2,20,1,1,1,1
            x_pred,y_pred = arima_pred(lista,parametros,INIT=len(lista)-30,FIM=len(lista)+30)
             
            der = np.array(lista)
            der = list(der[1:] - lista[:len(der)-1])
            der = np.array(der)
            
            ######################## BUY ###########################
            
            if ((y_pred[-20:].mean()-lista[-1]>5) or der[-1]<-20) and len(buy)<1:
                buy.append(last)
                #buy_btc(0.001)
                print('comprado por:',buy)
                
            ######################## SELL ##########################   
    
            if len(buy)>0: 
                if last-buy[0]>(10+last*(0.07/100)) or last<0.975*buy[0] :
                    history.append((last-buy[0])*(1-0.07/100))
                    sell.append(last)
                    buy.pop(0)
                    #sell_btc(0.001)
                    
            ######################## SHOW ##########################
            plot(lista,x_pred,y_pred)  
            print('arima derivate:',y_pred[-30:].mean()-float(resposta['last']))
            print('Preço atual:',float(resposta['last']))
            print('Comprado por:',buy)
            print('Total:',sum(history))
                
        else:
           wait = 61 - float(str(datetime.fromtimestamp(time.time())).split(':')[2])
           print('esperando:',wait,'segundos')
           time.sleep(wait)
    except:
        print('conection lost')
        time.sleep(10)


import statsmodels
statsmodels.tsa.stattools.acf(lista)
plt.plot(statsmodels.tsa.stattools.acf(lista))

lag = np.array(statsmodels.tsa.stattools.acf(lista))

lag[lag>0.2]
plt.plot(lag[lag>0.2])
'''


BTC_quantity = 0.00360 
WINDOW = ['M1','M3','M5','M15','M30','H1','H4','D1','W1']
horizon = 30
window = 300
timeframe = WINDOW[3]

def PF(horizon,window,timeframe):
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
    last = 0
    while 1:
        try:
            lista,timestamp=candles(WINDOW[4],window)
            lista = moving_average(lista, 4)
            #lista.append(last)
            
            #last = lista[-1]         
            if lista[-1] != last:
                last = lista[-1]
                
                #################### INDICATORS ########################
               
                #print('Runing Gaussian Process')
                #x_gpr,gpr_y_pred, y_std, = GPR(lista,660)
                #ETS_ypred = ETS.exponential_smooth(lista, 601, 660)
    
                der = np.array(lista)
                der = list(der[1:] - lista[:len(der)-1])
                der = np.array(der)   
                p = der/np.array(lista[:-1])
                print('Runing ARIMA')
                parameters = 1,1,1,1,2,1,horizon,1,1,1,1
                x_pred,y_pred,pred_ci = arima_pred(lista,parameters,INIT=len(lista),FIM=len(lista)+horizon)
                #x_pred,y_pred,pred_ci = arima_pred(x,parameters,INIT=len(x),FIM=len(lista)+horizon)
                x_pred,y_pred,pred_ci = arima_pred(p,parameters,INIT=len(p),FIM=len(p)+horizon)

                dydx = (y_pred[-horizon:].mean()-lista[-1])/(horizon/2)

                ######################## BUY ##########################
                
                plot(p,x_pred,y_pred,pred_ci[:,0],pred_ci[:,1],dydx,SELL,BUY)
                #plot(x,x_pred,y_pred,pred_ci[:,0],pred_ci[:,1],dydx,SELL,BUY)
                #plot(lista,x_gpr,gpr_y_pred,gpr_y_pred+y_std,gpr_y_pred-y_std) 
                #plot(lista,x_pred,ETS_ypred,ETS_ypred,ETS_ypred)
                print('arima derivate:',dydx)
                print('Preço atual:',lista[-1])
                print('Comprado por:',buy)
                print('Total:',sum(history))
                balance()
                deriv_hist.append((y_pred[-horizon:].mean()-lista[-1],monitor('timestamp')))    
                lista.append(last)      
            else:
                print('Esperando passagem do candle')
                time.sleep(5)
        except ConnectionError:
            print('ConnectionError')
            time.sleep(30)
            lista = []
            #lista.append(monitor('last'))

#PF(horizon,window,timeframe)




