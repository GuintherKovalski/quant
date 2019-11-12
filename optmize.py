import asyncio
import websockets.client
import json
import requests
import time
import sys
import hmac
import hashlib
import base64
from urllib import parse
import datetime
import gzip
import urllib
import time
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score 
import matplotlib.pyplot as plt
import requests

session = requests.session()
session.auth = ("cUH6YEYiZ0DkZHdMbu9bvCmyyxHi9ErS", "AHAhMcMwI41YbDar69likScAJXSYlRHO")
saldo = session.get('https://api.hitbtc.com/api/2/trading/balance').json()
BTC = float(saldo[76]['available'])
USD = float(saldo[503]['available'])
print('saldo BTC:',BTC)
print('saldo USD:',USD)

#vende o equivalente a um dólar em BTC, na cotação atual, instantaneo
def sell_btc():
    simbolo = 'BTCUSD'
    url = "https://api.hitbtc.com/api/2/public/ticker/"+simbolo
    r = requests.get(url)
    retorno = {}
    resposta = r.json()
    price = (float(resposta['low'])+float(resposta['high']))/2
    ORDER_SIZE_BTC = 0.0001 #BTC's
    simbolo = 'BTCUSD'
    session = requests.session()
    session.auth = ("cUH6YEYiZ0DkZHdMbu9bvCmyyxHi9ErS", "AHAhMcMwI41YbDar69likScAJXSYlRHO")
    orderData = {'symbol':simbolo, 'side': 'sell', 'quantity': str(ORDER_SIZE_BTC), 'price': str((price)*ORDER_SIZE_BTC) }
    r = session.post('https://api.hitbtc.com/api/2/order', data = orderData)        
    #print(r.json())

def buy_btc():
    simbolo = 'BTCUSD'
    url = "https://api.hitbtc.com/api/2/public/ticker/"+simbolo
    r = requests.get(url)
    retorno = {}
    resposta = r.json()
    price = (float(resposta['low'])+float(resposta['high']))/2
    ORDER_SIZE_USD = 0.0001*price
    simbolo = 'BTCUSD'
    session = requests.session()
    session.auth = ("cUH6YEYiZ0DkZHdMbu9bvCmyyxHi9ErS", "AHAhMcMwI41YbDar69likScAJXSYlRHO")
    orderData = {'symbol':simbolo,
                 'side': 'buy',
                 'quantity': str(0.0001),
                 'price': str(price) }
    r = session.post('https://api.hitbtc.com/api/2/order', data = orderData)        
    #print(r.json())
lista = []
a=0

def arima_pred(lista):
    colors = np.linspace(start=100, stop=255, num=90)
    p,d,q,P,D,Q,s,a1,a2,a3,a4 = 1,1,1,2,2,2,15,1,1,1,1
    train = np.array(lista)
    (int(p),int(d),int(q))
    param = (int(p),int(d),int(q))
    param_seasonal = (int(P),int(D),int(Q),int(s))
    TREND = [int(a1>0.5),int(a2>0.5),int(a3>0.5),int(a4>0.5)]
    train = np.array(lista)
    t=np.reshape(np.array(train).astype(float),-1,1)
    mod = sm.tsa.statespace.SARIMAX(t,order=param,seasonal_order=param_seasonal, enforce_stationarity=True,enforce_invertibility=True,initialization='approximate_diffuse')
    results = mod.fit()
    pred0 = results.get_prediction(start= 195,end = 230, dynamic=False)
    pred0_ci = pred0.conf_int()
    y_pred = pred0.predicted_mean
    x = np.array(range(195,231))
    return x,y_pred

sell = []
buy = []
history = []

while 1:
    simbolo = 'BTCUSD'
    url = "https://api.hitbtc.com/api/2/public/ticker/"+simbolo
    r = requests.get(url)
    retorno = {}
    resposta = r.json()
    if len((lista))<198:
        lista.append(float(resposta['last']))
        print(len(lista))
    elif lista[-1] != resposta['last']:
        a=a+1
        lista.append(float(resposta['last']))
        x,y_pred = arima_pred(lista)
        der = np.array(lista)
        der = list(der[1:] - lista[:len(der)-1])
        der = np.array(der)
        
        ######################## BUY ###########################
        print('comprado por:',buy)
        if (y_pred[-30:].mean()-float(resposta['last']))>8 and len(buy)<3:
            buy.append(float(resposta['last']))
            #buy_btc()
            
        ######################## SELL ##########################   

        if len(buy)>0: 
            if float(resposta['last'])-buy[0]>(10+float(resposta['last'])*(0.07/100)) or float(resposta['last'])<0.975*buy[0] :
                history.append((float(resposta['last'])-buy[0])*(1-0.07/100))
                sell.append(float(resposta['last']))
                buy.pop(0)
                #sell_btc()
                
        
        p = plt.figure(figsize=(6,4))
        plt.title('total:'+str(np.round(sum(history),2)))
        plt.style.use('classic') 
        plt.plot(x,y_pred)
        plt.plot(lista)
        plt.ylabel('USD/BTC')
        plt.xlabel('Sample')
        plt.xlim(0,230)
        plt.grid()
        plt.savefig('img/'+str(a)+'arima.jpg')
        plt.show()
        plt.clf()
        plt.close(p)
        
        print('arima derivate:',y_pred[-30:].mean()-float(resposta['last']))
        print('Preço atual:',float(resposta['last']))
        print('Comprado por:',buy)
        print('Historico:',history[-5:])
        print('Total:',sum(history))
    if len(lista)>200:
        lista.pop(0)






















