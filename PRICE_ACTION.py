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
plt.style.use('classic') 
    
def ALL():
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
    
    def vol(window,period):
        period = str(period)
        window = WINDOW[0]
        url = 'https://api.hitbtc.com/api/2/public/candles/BTCUSD?period=' + window+'&limit='+period
        r = requests.get(url)
        retorno = {}
        resposta = r.json()
        VOL,TIMESTAMP = [ ],[ ]
        for i in range(len(np.array(resposta))):
            VOL.append((float(np.array(resposta)[i]['volume'])))
            TIMESTAMP.append(np.array(resposta)[i]['timestamp'])
        return list(VOL), TIMESTAMP
    
    from scipy.signal import savgol_filter
    WINDOW = ['M1','M3','M5','M15','M30','H1','H4','D1','W1']
    
    window = 1000
    lista,timestamp=candles(WINDOW[0],window)
    w = 13
    #x = moving_average(lista, w)
    x = savgol_filter(lista, 63, 3)
    #x = savgol_filter(lista, 123, 3)
    d_window = 10
    dydx = x[1:]-x[:-1]
    
    val = []
    val.append(0)
    index = []
    l=130
    j = 190
    for i in range(0,len(lista)-l,l):
        a = min(lista[i:i+j])
        if a!= val[-1]:
            index.append(int(np.argwhere(np.array(lista)==a)[0]))
            val.append(a)
    val.pop(0)
    a = min(lista[-80:])  
    val.append(a)
    index.append(int(np.argwhere(np.array(lista)==a)))
    y_s=val
    x_s=index
    
    val = []
    val.append(0)
    index = []
    for i in range(0,len(lista)-l,l):
        a = max(lista[i:i+j])
        if a!= val[-1]:
            index.append(int(np.argwhere(np.array(lista)==a)[0]))
            val.append(a)
    val.pop(0)
    a = max(lista[-80:])  
    val.append(a)
    index.append(int(np.argwhere(np.array(lista)==a)))
    
    
    '''
    indexMAX = []
    indexMIN = []
    for i in range(len(x)-1):
        if dydx[i]<0 and sum(dydx[i+1:i+50]>0)==49:
            indexMAX.append(i)
        if dydx[i]>0 and sum(dydx[i+1:i+50]<0)==49:
            indexMIN.append(i)
           
    #plt.scatter(  np.array(indexMIN)  , np.array(lista)[np.array(indexMIN)]   )   
    #plt.plot(x) 
    #plt.plot(lista[w:])  
    y_s = np.array(lista)[list(np.array(indexMIN)+int(w/2))]
    x_s = np.array(indexMIN)+w/2
    #A = (y_s[0]-y_s[1])/(x_s[0]-x_s[1]) 
    #B = y_s[0]-A*x_s[0]
    #X_s=np.array(range(-10,200))
    #Y_s = A*X+B
    y_r = np.array(lista)[list(np.array(indexMAX)+int(w/2))]
    x_r = np.array(indexMAX)+w/2
    
    A = (y_r[0]-y_r[1])/(x_r[0]-x_r[1]) 
    B = y_r[0]-A*x_r[0]
    X_r = np.array(range(-10,200))
    Y_r = A*X+B
    '''
    y_r=val
    x_r=index
    
    plt.figure(figsize= (10,5))
    plt.plot(lista) 
    
    c_gap =1
    a_r = []
    a_s = []
    
    for i in range(0,len(y_r)-c_gap,c_gap):
        A = (y_r[i]-y_r[i+c_gap])/(x_r[i]-x_r[i+c_gap]+0.001) 
        B = y_r[i]-A*x_r[i]
        a_r.append(A)
        X_r1 = np.array(range(int(x_r[i]),int(x_r[i+c_gap])))
        Y_r1 = A*X_r1+B
        #Y_r1 = A*X+B
        plt.plot(X_r1,Y_r1,color = 'green')
        if i == list(range(0,len(y_r)-c_gap,c_gap))[-1]:  
            X_r1 = np.array(range(int(x_r[i]),int(len(lista)+60)))
            Y_r1 = A*X_r1+B
            plt.plot(X_r1,Y_r1,color = 'green')
    c_gap =1
    for i in range(0,len(y_r)-c_gap,c_gap):
        try:
            A = (y_s[i]-y_s[i+c_gap])/(x_s[i]-x_s[i+c_gap]) 
            a_s.append(A)
            B = y_s[i]-A*x_s[i]
            X_s = np.array(range(int(x_s[i]),int(x_s[i+c_gap])))
            Y_s = A*X_s+B
            plt.plot(X_s,Y_s,color = 'red') 
            if i == list(range(0,len(y_r)-c_gap,c_gap))[-1]:
                X_s = np.array(range(int(x_s[i]),int(len(lista)+60)))
                Y_s = A*X_s+B
                plt.plot(X_s,Y_s,color = 'red') 
        except:
            i = len(x_s)-1-c_gap
            A = (y_s[i]-y_s[i+c_gap])/(x_s[i]-x_s[i+c_gap]) 
            a_s.append(A)
            B = y_s[i]-A*x_s[i]      
            X_s = np.array(range(int(x_s[i]),int(len(lista)+60)))
            Y_s = A*X_s+B
            plt.plot(X_s,Y_s,color = 'red') 
            
    plt.plot(x) 
    plt.grid()
    plt.savefig('dydx/Sup_Res'+str(time.time()).split('.')[0]+'.png')  
    plt.title('Support/Resistence')
    plt.xlim(0,len(lista)+60)
    plt.savefig('../INDICATORS/'+'support_res'+str(time.time())+'.png')
    plt.cla()
    plt.clf()
    
    #plt.plot(X_r,Y_r)
    #plt.plot(X_s,Y_s)
    #plt.plot(x) 
    #plt.plot(lista)
    #plt.xlim(20,115)  
    #plt.ylim(min(lista)-20,max(lista)+20)
    #plt.legend(['MA 10 periodos','preço'])
    #plt.figure(1)
    #plt.plot(a_s)
    #plt.plot(a_r)
    
    price = np.array(lista)
    price  = (price-price.mean())
    price = price/max(price)
    dydx = savgol_filter(dydx, 63, 3)
    price = savgol_filter(price, 63, 3)
    
    plt.figure(66)
    plt.figure(figsize= (10,4))
    #plt.figure(3,figsize=(6,2))
    plt.plot(dydx/max(abs(dydx)))
    plt.plot(price)
    plt.plot(np.zeros(len(dydx)))
    plt.legend(['dydx','price'],loc='best')
    plt.savefig('dydx/Derivate'+str(time.time()).split('.')[0]+'.png')  
    plt.title('Derivate')
    plt.savefig('../INDICATORS/'+'phase_in_t'+str(time.time())+'.png')
    plt.cla()
    plt.clf()
    
    
    
    r=int(-1)
    s=int(-1)
    plt.figure(50)
    plt.figure(figsize= (10,5))
    cem = y_r[r]-y_s[s]
    fibonacci = np.ones((400,5))*np.array([0.236,0.382,0.50,0.618,0.786])*cem+y_s[-1]
    x_f = list(range(x_s[s],x_s[s]+len(fibonacci),1))
    #plt.plot(x) 
    plt.plot(lista) 
    plt.plot(x_f,fibonacci)
    plt.ylim(y_r[r]-200,y_s[s]+200)
    plt.legend(['price','0.236','0.382','0.50','0.618','0.786'])
    plt.grid()
    plt.title('Fibonacci')
    plt.savefig('dydx/Fibonacci'+str(time.time()).split('.')[0]+'.png')  
    plt.savefig('../INDICATORS/'+'Fibonacci_'+str(time.time())+'.png')
    
    import numpy
    import talib
    output = talib.SMA(np.array(lista))
    from talib import MA_Type
    upper, middle, lower = talib.BBANDS(np.array(lista),44)
    #talib.MOM(lista, timeperiod=5)
    
    plt.figure(60)
    plt.figure(figsize= (10,5))
    plt.plot(upper)
    plt.plot(lower)
    plt.plot(middle)
    plt.plot(np.array(lista))
    plt.grid()
    volume,timestamp=vol(WINDOW[0],window)
    plt.legend(['upper','lower','mid','price'])
    plt.title('Bollinger')
    plt.savefig('../INDICATORS/'+'dydxBollinger'+str(time.time()).split('.')[0]+'.png')  
    plt.cla()
    plt.clf()
    
    price = np.array(lista)
    std = np.ones(len(volume))*np.array(volume).std()
    mean = np.ones(len(volume))*np.array(volume).mean()
    
    plt.figure(70)
    plt.figure(figsize= (10,5))
    plt.plot(volume)
    plt.plot(mean+2*std)
    plt.plot((price-(min(price))))
    plt.grid()
    plt.legend(['volume','mean+2sigma','price (normalized)'])
    plt.title('Anomaly')
    plt.savefig('dydx/anomaly'+str(time.time()).split('.')[0]+'.png')  
    plt.savefig('../INDICATORS/'+'dydx_'+str(time.time())+'.png')
    plt.cla()
    plt.clf()
    
    
    plt.figure(777)
    #plt.figure(figsize= (5,5))
    dyd2x = dydx[1:]-dydx[:-1]
    state1 = (dydx/max(abs(dydx)))[1:]
    state2 = dyd2x
    lenght = 200
    state2 = state2[-lenght:]
    state1 = state1[-lenght:] 
    plt.grid()
    t = np.linspace(0, 1, num=len(state1))
    dotcolors=[(0.0, 0.0, 0.0, a) for a in t]
    plt.scatter(state1, state2, c=dotcolors, s=30, edgecolors='None')
    plt.plot(state1,state2,linewidth=0.4,c ='black')
    [min(state2),max(state2)]
    [min(state1),max(state2)]
    [0,0]
    plt.plot([min(state1),max(state1)],[0,0],color = 'black')
    plt.plot([0,0],[min(state2),max(state2)],color = 'black')
    plt.title('Phase Diagram')
    plt.savefig('../INDICATORS/'+'phase_mean'+str(time.time())+'.png')
    
    plt.xlabel("y'(x)")
    plt.ylabel("y''(x)")
    plt.cla()
    plt.clf()
    
    
    plt.figure(999)
    price = np.array(lista)
    price  = (price-price.mean())
    price = price/max(price)
    price = savgol_filter(price, 63, 3)
    #plt.figure(figsize= (5,5))
    dyd2x = dydx[1:]-dydx[:-1]
    state1 = (dydx/max(abs(dydx)))[1:]
    state2 = price
    lenght = 300
    state2 = state2[-lenght:]
    state1 = state1[-lenght:] 
    plt.grid()
    t = np.linspace(0, 1, num=len(state1))
    dotcolors=[(0.0, 0.0, 0.0, a) for a in t]
    plt.scatter(state2, state1, c=dotcolors, s=30, edgecolors='None')
    plt.plot(state2,state1,linewidth=0.4,c ='black')
    plt.title("Phase Diagram price y'(x)")
    plt.xlabel("x")
    plt.ylabel("y'(x)")
    plt.savefig('../INDICATORS/'+'phase'+str(time.time())+'.png')
    
    [min(state2),max(state2)]
    [min(state1),max(state2)]
    [0,0]
    plt.plot([0,0],[min(state1),max(state1)],color = 'black')
    plt.plot([min(state2),max(state2)],[0,0],color = 'black')
    plt.cla()
    plt.clf()
    
    #plt.scatter(dydx/max(abs(dydx)),price[1:])
    #kernel = (dydx/max(abs(dydx)))*price[1:]
    #plt.plot(kernel)
