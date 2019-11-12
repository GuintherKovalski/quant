import matplotlib.pyplot as plt
import talib
import numpy as np
'''
Mais indicadores e o significado destes em:
    https://mrjbq7.github.io/ta-lib/doc_index.html
'''

def features(BITCOIN):
    BITCOIN['MEAN'] = (BITCOIN.low+BITCOIN.high)/2
    NATR =         talib.NATR(BITCOIN.high, BITCOIN.low, BITCOIN.close)
    HT_DCPERIOD =  talib.HT_DCPERIOD(BITCOIN.close)
    ADOSC =        talib.ADOSC(BITCOIN.high, BITCOIN.low, BITCOIN.close, BITCOIN.volume, fastperiod=3, slowperiod=10)
    RSI =          talib.RSI(BITCOIN.MEAN, timeperiod=10)
    ROCP =         talib.ROCP(BITCOIN.MEAN, timeperiod=10)
    SLOWK, SLOWD = talib.STOCH(BITCOIN.high, BITCOIN.low, BITCOIN.close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    WILLR =        talib.WILLR(BITCOIN.high, BITCOIN.low, BITCOIN.close, timeperiod=10)
    ULTOSC =       talib.ULTOSC(BITCOIN.high, BITCOIN.low, BITCOIN.close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    MACD, MACDSIGNAL, MACDHIST = talib.MACD(BITCOIN.close, fastperiod=10, slowperiod=20, signalperiod=9) 
   
    NAN = []
    NAN.append(sum(np.isnan(np.array(NATR))))
    NAN.append(sum(np.isnan(np.array(HT_DCPERIOD))))
    NAN.append(sum(np.isnan(np.array(ADOSC))))
    NAN.append(sum(np.isnan(np.array(RSI))))
    NAN.append(sum(np.isnan(np.array(SLOWK))))
    NAN.append(sum(np.isnan(np.array(SLOWD))))
    NAN.append(sum(np.isnan(np.array(ULTOSC))))
    NAN.append(sum(np.isnan(np.array(WILLR))))
    NAN.append(sum(np.isnan(np.array(MACD))))
    NAN.append(sum(np.isnan(np.array(MACDSIGNAL))))
    max(NAN)
    
    NATR[np.isnan(np.array(NATR))] = 0
    HT_DCPERIOD[np.isnan(np.array(HT_DCPERIOD))] = 0
    ADOSC[np.isnan(np.array(ADOSC))] = 0
    RSI[np.isnan(np.array(RSI))] = 0
    ROCP[np.isnan(np.array(ROCP))] = 0
    SLOWD[np.isnan(np.array(SLOWD))] = 0
    SLOWK[np.isnan(np.array(SLOWK))] = 0
    WILLR[np.isnan(np.array(WILLR))] = 0
    ULTOSC[np.isnan(np.array(ULTOSC))] = 0
    MACDSIGNAL[np.isnan(np.array(MACDSIGNAL))] = 0
    MACD[np.isnan(np.array(MACD))] = 0
    
    BTC = np.array(BITCOIN.volume)
    V_DERIVATE = []
    V_DERIVATE = list(BTC[1:] - BTC[:len(BITCOIN.iloc[:,0])-1])
    #DERIVATE.insert(0,0)
    V_DERIVATE.insert(0,0)
    BTC = np.array(BITCOIN.MEAN)
    M_DERIVATE = []
    M_DERIVATE = list(BTC[1:] - BTC[:len(BITCOIN.iloc[:,0])-1])
    #DERIVATE.insert(0,0)
    M_DERIVATE.insert(0,0)
        
    BITCOIN['MEAN_Derivate'] = np.array(M_DERIVATE)
    BITCOIN['Volume_Derivate'] = np.array(V_DERIVATE) 
    BITCOIN['NATR'] = NATR
    BITCOIN['HT_DCPERIOD'] = HT_DCPERIOD
    BITCOIN['ADOSC'] = ADOSC
    BITCOIN['RSI'] = RSI
    BITCOIN['ROCP'] = ROCP
    BITCOIN['SLOWD'] = SLOWD
    BITCOIN['WILLR'] = WILLR
    BITCOIN['ULTOSC'] = ULTOSC
    BITCOIN['MACD'] = MACD
    BITCOIN['MACDSIGNAL'] = MACDSIGNAL
    
    BITCOIN = BITCOIN.iloc[max(NAN):,:] #alguns indicadores precisam de 32 amostras pra serem gerados
    
    return BITCOIN
   

'''
BITCOIN =      DATA
    NATR =         talib.NATR(BITCOIN.high, BITCOIN.low, BITCOIN.close)
    HT_DCPERIOD =  talib.HT_DCPERIOD(BITCOIN.close)
    AVGPRICE =     talib.AVGPRICE(BITCOIN.open, BITCOIN.high, BITCOIN.low, BITCOIN.close)
    ADOSC =        talib.ADOSC(BITCOIN.high, BITCOIN.low, BITCOIN.close, BITCOIN.volume, fastperiod=3, slowperiod=10)
    OBV =          talib.OBV(BITCOIN.close, BITCOIN.volume)
    RSI =          talib.RSI(BITCOIN.close, timeperiod=14)
    ROC =          talib.ROC(BITCOIN.close, timeperiod=10)
    ROCP =         talib.ROCP(BITCOIN.close, timeperiod=10)
    SLOWK, SLOWD = talib.STOCH(BITCOIN.high, BITCOIN.low, BITCOIN.close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    WILLR =        talib.WILLR(BITCOIN.high, BITCOIN.low, BITCOIN.close, timeperiod=14)
    ULTOSC =       talib.ULTOSC(BITCOIN.high, BITCOIN.low, BITCOIN.close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    MACD, MACDSIGNAL, MACDHIST = talib.MACD(BITCOIN.close, fastperiod=12, slowperiod=26, signalperiod=9) 
    
    NATR[np.isnan(np.array(NATR))] = 0 
    HT_DCPERIOD[np.isnan(np.array(HT_DCPERIOD))] = 0
    AVGPRICE[np.isnan(np.array(AVGPRICE))] = 0 
    ADOSC[np.isnan(np.array(ADOSC))] = 0
    OBV[np.isnan(np.array(OBV))] = 0
    RSI[np.isnan(np.array(RSI))] = 0
    ROC[np.isnan(np.array(ROC))] = 0
    ROCP[np.isnan(np.array(ROCP))] = 0
    SLOWD[np.isnan(np.array(SLOWD))] = 0
    WILLR[np.isnan(np.array(WILLR))] = 0
    ULTOSC[np.isnan(np.array(ULTOSC))] = 0
    MACDSIGNAL[np.isnan(np.array(MACDSIGNAL))] = 0
    MACD[np.isnan(np.array(MACD))] = 0
    
    BITCOIN = BITCOIN.iloc[:,[2,7]]
    
    BTC = np.array(BITCOIN.open)
    DERIVATE = []
    DERIVATE = list(BTC[1:] - BTC[:len(BITCOIN.iloc[:,0])-1])
    DERIVATE.insert(0,0)
    BITCOIN['open derivate'] = np.array(DERIVATE) 
    
    BTC = np.array(BITCOIN.volume)
    DERIVATE = []
    DERIVATE = list(BTC[1:] - BTC[:len(BITCOIN.iloc[:,0])-1])
    #DERIVATE.insert(0,0)
    DERIVATE.insert(0,0)
    BITCOIN['Volume_Derivate'] = np.array(DERIVATE) 
    
    BITCOIN['NATR'] = NATR
    BITCOIN['HT_DCPERIOD'] = HT_DCPERIOD
    BITCOIN['ADOSC'] = ADOSC
    BITCOIN['RSI'] = RSI
    BITCOIN['ROCP'] = ROCP
    BITCOIN['SLOWD'] = SLOWD
    BITCOIN['WILLR'] = WILLR
    BITCOIN['ULTOSC'] = ULTOSC
    BITCOIN['MACD'] = MACD
    BITCOIN['MACDSIGNAL'] = MACDSIGNAL
    
    BITCOIN = BITCOIN.iloc[32:,:] #alguns indicadores precisam de 32 amostras pra serem gerados
    BITCOIN.to_csv('BTC.csv')  
'''



