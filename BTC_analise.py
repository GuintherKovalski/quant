

window = 1000
lista,timestamp=candles(WINDOW[4],window)

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
smoth = moving_average(lista, 15)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

y = np.array(lista)
dfy = pd.DataFrame(y, columns = ['value'])
print(pd.DataFrame(y).describe())
"""
Estatisticas basicas
"""

"""
Plota a decomposição em series de fourier,
com o periodo invés de a frequência
"""

signal = y
fourier = np.fft.rfft(signal)
n = signal.size
sample_rate = 1
freq = np.fft.fftfreq(n, d=1./sample_rate)
module= (np.real(y)**2+np.imag(y)**2)**0.5
angle= np.angle(y)
plt.figure(figsize=(60,4))   
periodo=1/freq[:int(len(module)/2)]
plt.plot(module[:int(len(module)/2)])
#plt.plot(angle[:int(len(angle)/2)])
plt.grid(linestyle='dashed')
plt.ylim(min(module[:int(len(module)/2)]),max(module[:int(len(module)/2)]))
plt.xlim(1,82)
plt.xticks(range(0,len(periodo),5), periodo[range(0,len(periodo),5)].round(decimals = 2)) 
periodo=1/freq[:int(len(module)/2)]
plt.xlabel("Period (1/F)")
plt.ylabel("Amplitude")
index = np.argwhere(max(module[:int(len(module)/2)])==module[:int(len(module)/2)])
print('period of max amp:',periodo[int(index)],'max amp:', module[index])
plt.savefig('fourier_spectrum.eps', format='eps', dpi=1000)

"""
Plota a autocorrelação da série
"""
from pandas.plotting import autocorrelation_plot
plt.figure(figsize=(16,4))  
autocorrelation_plot(y)
plt.grid(linestyle='dashed')
plt.ylim(-1,1)
plt.xlim(1,len(y))
#plt.xticks(range(0,len(periodo),20), periodo[range(0,len(periodo),20)]) 
#periodo=1/freq[:int(len(module)/2)]
plt.xlabel("Período")
plt.ylabel("Amplitude")
plt.savefig('autocorr.eps', format='eps', dpi=1000)


"""
Plota a série original, sazonalidade e tendencia
"""
decomposition = seasonal_decompose(y,freq=29)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(16,4))  
plt.grid(linestyle='dashed')
plt.plot(y, label='Original',color='red')
plt.title("Original")
plt.ylabel("Amplitude")
plt.xlabel("Samples (weeks)")
plt.savefig('Original.eps', format='eps', dpi=1000)

plt.figure(figsize=(16,4)) 
plt.grid(linestyle='dashed') 
plt.plot(trend, label='Trend',color='red')
plt.title("Trend")
plt.ylabel("Amplitude")
plt.xlabel("Samples (weeks)")
plt.savefig('trend.eps', format='eps', dpi=1000)

plt.figure(figsize=(16,4)) 
plt.grid(linestyle='dashed')
plt.plot(seasonal,label='Seasonality',color='green')
plt.title("Seasonality")
plt.ylabel("Amplitude")
plt.xlabel("Samples (weeks)")
plt.savefig('Seasonal.eps', format='eps', dpi=1000)

plt.figure(figsize=(16,4)) 
plt.grid(linestyle='dashed')
plt.plot(residual,label='Seasonality',color='green')
plt.title("Residuals")
plt.ylabel("Amplitude")
plt.xlabel("Samples (weeks)")
plt.savefig('Seasonal.eps', format='eps', dpi=1000)

"""
Teste de Estacionariedade
"""
from statsmodels.tsa.stattools import adfuller
dfytest = adfuller(y, autolag='AIC')
dfyoutput = pd.Series(dfytest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
print(dfyoutput)

x = seasonal
x = x[np.logical_not(np.isnan(x))]
dfytest1 = adfuller(x, autolag='AIC')
dfyoutput1 = pd.Series(dfytest1[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
print('Dickey Fuller Test:\n',dfyoutput1)


"""
Filtro de Outlaiers
"""
index_of_non_outlaiers=[]
for i in range(y.shape[0]):
    delete = []
    std=y.std()
    m=y.mean()
    first_list = list(np.reshape(np.argwhere(y<m+2*std),-1,1))
    second_list = list(np.reshape(np.argwhere(y>m-2*std),-1,1))
    index_of_non_outlaiers.append(list(set(first_list).intersection(second_list)))
    
"""
Plota a série com filtros de lowess
"""
from statsmodels.nonparametric.smoothers_lowess import lowess
plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

dfy_loess_5 = pd.DataFrame(lowess(y, np.arange(len(y)), frac=0.05)[:, 1], index=dfy.index, columns=['value'])
dfy_loess_15 = pd.DataFrame(lowess(y, np.arange(len(y)), frac=0.15)[:, 1], index=dfy.index, columns=['value'])
dfy_loess_30 = pd.DataFrame(lowess(y, np.arange(len(y)), frac=0.3)[:, 1], index=dfy.index, columns=['value'])
fig, axes = plt.subplots(4,1, figsize=(7, 7), sharex=True, dpi=120)
dfy['value'].plot(ax=axes[0], color='k', title='Original Series')
dfy_loess_5['value'].plot(ax=axes[1], title='Loess Smoothed 5%')
dfy_loess_15['value'].plot(ax=axes[2], title='Loess Smoothed 15%')
dfy_loess_30['value'].plot(ax=axes[3], title='Loess Smoothed 30%')


"""
Plota a série com filtros de lowess
"""
plt.figure(figsize=(16,4)) 
dfy_ma = dfy.rolling(50, center=True)
plt.plot(dfy_ma.mean())
fig.suptitle('How to Smoothen a Time Series', y=0.95, fontsize=14)
plt.show()

"""
Autocorrelation by lags
"""
from pandas.plotting import lag_plot
plt.rcParams.update({'ytick.left' : False, 'axes.titlepad':10})

# Import
ss = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/sunspotarea.csv')
a10 = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv')

# Plot
fig, axes = plt.subplots(1, 4, figsize=(10,3), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(dfy.value, lag=i+1, ax=ax, c='firebrick')
    ax.set_title('Lag ' + str(i+1))
fig.suptitle('Lag Plots of Sun Spots Area \n(Points get wide and scattered with increasing lag -> lesser correlation)\n', y=1.15)    


"""
Partial Autocorrelation 
"""

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#dfy = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv')
# Calculate ACF and PACF upto 50 lags
# acf_50 = acf(dfy.value, nlags=50)
# pacf_50 = pacf(dfy.value, nlags=50)
# Draw Plot
fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(dfy.value.tolist(), lags=30, ax=axes[0])
plot_pacf(dfy.value.tolist(), lags=30, ax=axes[1])

'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dfy_scaled = pd.DataFrame(scaler.fit_transform(dfy), columns=dfy.columns)
x = dfy_scaled
#dfy_scaled.to_csv('mim_max.csv')
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
plt.scatter(principalComponents[:,0],principalComponents[:,1])
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(principalComponents)
cluster1 = np.argwhere(kmeans.labels_==1)
cluster2 = np.argwhere(kmeans.labels_==0)
plt.figure(50)
c1 = principalComponents[cluster1,:]
c2 = principalComponents[cluster2,:]
C1 = pca.inverse_transform(c1) 
C2 = pca.inverse_transform(c2) 
m = principalComponents[cluster1,:].mean()
std = principalComponents[cluster1,:].std()
anomalies = principalComponents[cluster1,:] > m + 3*std
plt.scatter(principalComponents[cluster1,0],principalComponents[cluster1,1])
plt.scatter(principalComponents[cluster2,0],principalComponents[cluster2,1])
plt.scatter(kmeans.cluster_centers_[0,0],kmeans.cluster_centers_[0,1])
plt.scatter(kmeans.cluster_centers_[1,0],kmeans.cluster_centers_[1,1])
'''


import os
import pywt
#from wavelets.wave_python.waveletFunctions import *
import itertools
import numpy as np
import pandas as pd
from scipy.fftpack import fft
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches


dataset = "https://raw.githubusercontent.com/taspinar/siml/master/datasets/sst_nino3.dat.txt"
#df_nino = BITCOIN.iloc[670472:,0]
nino = (np.array(lista)-np.array(lista).min())/(np.array(lista).max()-np.array(lista).min())
df_nino = pd.DataFrame(nino,columns = ['values']) #pd.read_table(dataset)

N = df_nino.shape[0]
t0=2015
dt=8/signal.shape[0]

signal = df_nino# DATA[:,1]*2
time = np.arange(0, signal.shape[0]) * dt + t0
signal = df_nino.values.squeeze()



# First lets load the el-Nino dataset, and plot it together with its time-average

def get_ave_values(xvalues, yvalues, n = 5):
    signal_length = len(xvalues)
    if signal_length % n == 0:
        padding_length = 0
    else:
        padding_length = n - signal_length//n % n
    xarr = np.array(xvalues)
    yarr = np.array(yvalues)
    xarr.resize(signal_length//n, n)
    yarr.resize(signal_length//n, n)
    xarr_reshaped = xarr.reshape((-1,n))
    yarr_reshaped = yarr.reshape((-1,n))
    x_ave = xarr_reshaped[:,0]
    y_ave = np.nanmean(yarr_reshaped, axis=1)
    return x_ave, y_ave

def plot_signal_plus_average(ax, time, signal, average_over = 5):
    time_ave, signal_ave = get_ave_values(time, signal, average_over)
    ax.plot(time, signal, label='signal')
    ax.plot(time_ave, signal_ave, label = 'time average (n={})'.format(5))
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylabel('Amplitude', fontsize=16)
    ax.set_title('Signal + Time Average', fontsize=16)
    ax.legend(loc='upper right')

fig, ax = plt.subplots(figsize=(12,3))
plot_signal_plus_average(ax, time, signal, average_over = 15)
plt.show()


def get_fft_values(y_values, T, N, f_s):
    N2 = 2 ** (int(np.log2(N)) + 1) # round up to next highest power of 2
    f_values = np.linspace(0.0, 1.0/(2.0*T), N2//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N2 * np.abs(fft_values_[0:N2//2])
    return f_values, fft_values

def plot_fft_plus_power(ax, time, signal, plot_direction='horizontal', yticks=None, ylim=None):
    dt = time[1] - time[0]
    N = len(signal)
    fs = 1/dt
    
    variance = np.std(signal)**2
    f_values, fft_values = get_fft_values(signal, dt, N, fs)
    fft_power = variance * abs(fft_values) ** 2
    if plot_direction == 'horizontal':
        ax.plot(f_values, fft_values, 'r-', label='Fourier Transform')
        ax.plot(f_values, fft_power, 'k--', linewidth=1, label='FFT Power Spectrum')
    elif plot_direction == 'vertical':
        scales = 1./f_values
        scales_log = np.log2(scales)
        ax.plot(fft_values, scales_log, 'r-', label='Fourier Transform')
        ax.plot(fft_power, scales_log, 'k--', linewidth=1, label='FFT Power Spectrum')
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels(yticks)
        ax.invert_yaxis()
        ax.set_ylim(ylim[0], -1)
    ax.legend()

fig, ax = plt.subplots(figsize=(12,3))
ax.set_xlabel('Frequency [Hz / year]', fontsize=18)
ax.set_ylabel('Amplitude', fontsize=18)
plot_fft_plus_power(ax, time, signal)
plt.show()


def plot_wavelet(ax, time, signal, scales, waveletname = 'cmor', 
                 cmap = plt.cm.seismic, title = '', ylabel = '', xlabel = ''):
    
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)
    
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
    
    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    
    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)
    return yticks, ylim

scales = np.arange(1, 128)
title = 'Wavelet Transform (Power Spectrum) of signal'
ylabel = 'Period (years)'
xlabel = 'Time'

fig, ax = plt.subplots(figsize=(10, 10))
plot_wavelet(ax, time, signal, scales, xlabel=xlabel, ylabel=ylabel, title=title)
plt.show()

fig = plt.figure(figsize=(12,12))
spec = gridspec.GridSpec(ncols=6, nrows=6)
top_ax = fig.add_subplot(spec[0, 0:5])
bottom_left_ax = fig.add_subplot(spec[1:, 0:5])
bottom_right_ax = fig.add_subplot(spec[1:, 5])

plot_signal_plus_average(top_ax, time, signal, average_over = 3)
yticks, ylim = plot_wavelet(bottom_left_ax, time, signal, scales, xlabel=xlabel, ylabel=ylabel, title=title)

plot_fft_plus_power(bottom_right_ax, time, signal, plot_direction='vertical', yticks=yticks, ylim=ylim)
bottom_right_ax.set_ylabel('Period [years]', fontsize=14)
plt.tight_layout()
plt.show()















