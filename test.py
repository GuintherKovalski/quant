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
lista = []
a=0

def arima_pred(lista):
    colors = np.linspace(start=100, stop=255, num=90)
    p,d,q,P,D,Q,s,a1,a2,a3,a4 = 1,1,1,2,2,2,2,1,1,1,1
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
    time.sleep(1)
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
        if (y_pred[-30:].mean()-float(resposta['last']))<15 and len(buy)<3:
            buy.append(float(resposta['last'])) 
            
        ######################## SELL ##########################   
        #if (np.array(der)[-10:].mean())<-1.1 and len(buy)>0:
        if len(buy)>0: 
            if float(resposta['last'])-buy[0]>(5+float(resposta['last'])*(0.07/100)) or float(resposta['last'])<0.99*buy[0] :
                history.append((float(resposta['last'])-buy[0])*(1-0.07/100))
                sell.append(float(resposta['last']))
                buy.pop(0)
                
        
        plt.figure(figsize=(6,4))
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
        
        print('arima derivate:',y_pred[-30:].mean()-float(resposta['last']))
        print('Preço atual:',float(resposta['last']))
        print('Comprado por:',buy)
        print('Historico:',history[-5:])
        print('Total:',sum(history))
    if len(lista)>200:
        lista.pop(0)
    
 
'''   
    
#plt.plot(pred0.predicted_mean)
x = np.array(range(41,60))
plt.title('Previsão da quantidade de chamadas - Comercial')
plt.plot(x,y_pred)
plt.plot(y)
plt.xlim(41,60)
plt.legend(['Previsões','Quantidade de chamadas'])
plt.xticks(np.array(lista)[41:60,0])    
       
async def hello():
    async with websockets.connect(url) as websocket:
        while 1: 
            await websocket.send(requisicao)
            data_report = await websocket.recv()  # Retorna True
            resposta_report = json.loads(data_report)


import matplotlib.pyplot as plt
import numpy

hl, = plt.plot([], [])

def update_line(hl, new_data):
    hl.set_xdata(numpy.append(hl.get_xdata(), new_data))
    hl.set_ydata(numpy.append(hl.get_ydata(), new_data))
    plt.draw()



simbolo = 'BTCUSD'

import asyncio
import websockets

async def hello():
    url = "https://api.hitbtc.com/api/2/public/ticker/"+simbolo
    async with websockets.connect(uri) as websocket:
        await websocket.recv()

r = asyncio.get_event_loop().run_until_complete(hello())


simbolo = 'BTCUSD'
url = "https://api.hitbtc.com/api/2/public/ticker/"+simbolo



        
import websocket 
import json
from websocket import create_connection
ws = create_connection("wss://ws.dogechain.info/inv")
ws.send(json.dumps({"op":"addr_sub", "addr":"dogecoin_address"}))
result =  ws.recv()
print (result)
ws.close()










import time
import queue
from hitbtc import HitBTC
c = HitBTC()
c.start()  # start the websocket connection
time.sleep(2)  # Give the socket some time to connect
c.subscribe_ticker(symbol='ETHBTC') # Subscribe to ticker data for the pair ETHBTC

while True:
    try:
        data = c.recv()
    except queue.Empty:
        continue

    # process data from websocket
    ...

c.stop()

'''

















