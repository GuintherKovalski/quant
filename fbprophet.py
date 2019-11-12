import math
import fbprophet
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.special import inv_boxcox
from matplotlib.ticker import StrMethodFormatter
from fbprophet.diagnostics import cross_validation

lista,timestamp=candles(WINDOW[3],600)

df = pd.DataFrame(lista,columns=['value'])
df['date'] = timestamp
df['date'] = pd.to_datetime(df['date'] , errors='coerce')
# Create new ds column from date
df['ds'] = df['date']
# Create new y column from value
df['y'] = df['value']
df['ds'] =  df['date']
df['ds'] = df['ds'].dt.tz_convert(None)
# Repurpose date column to be used as dataframe index
df = df.set_index("date")

df.tail()

# Create figure and subplot
plt.figure(figsize=(11,8))
ax = plt.subplot(111)

# Plot
df['value'].plot(color='#334f8d', fontsize=11, zorder=2, ax=ax)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Remove x-axis label
ax.set_xlabel('')

# Switch off ticks
ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

# Annotate
x_line_annotation = dt.datetime(2017, 5, 1) 
x_text_annotation = dt.datetime(2017, 5, 5)
ax.axvline(x=x_line_annotation, linestyle='dashed', alpha=0.5)
ax.text(x=x_text_annotation, y=670000, s='Trend Changepoint', alpha=0.7, color='#334f8d')

# Get y-axis tick values
vals = ax.get_yticks()

# Draw horizontal axis lines
for val in vals:
    ax.axhline(y=val, linestyle='dashed', alpha=0.3, color='#eeeeee', zorder=1)

# Format y-axis label
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

# Set y-axis label
ax.set_ylabel("Orders", labelpad=20, weight='bold')

# Set y-axis limit
ylim = ax.set_ylim(bottom=0)


# Apply Box-Cox Transform to value column
df['y'], lam = boxcox(df['value'])

# Print lambda value
print('Lambda: {}'.format(lam))

ax = df[['value','y']].plot(color='#334f8d', subplots=True, sharex=True, fontsize=11, legend=False, figsize=(11,12), title=['Untransformed','Box-Cox Transformed'])

for i, x in enumerate(ax):
    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)
    x.spines['bottom'].set_visible(False)

    # Remove x-axis label
    x.set_xlabel('')

    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # Format y-axis ticks
    vals = x.get_yticks()
    x.set_yticklabels(['{:,}'.format(int(y)) for y in vals])

    # Draw horizontal axis lines
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.3, color='#eeeeee', zorder=1)
        
    # Format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    # Set y-axis limit
    if i == 0:
      x.set_ylim(bottom=0)
      
      
# Instantiate Prophet object
m = fbprophet.Prophet()

# Call fit method on Prophet object and pass in prepared dataframe. This is where model fitting is performed
m.fit(df)

# Create a dataframe with ds extending the chosen number of periods into the future
future = m.make_future_dataframe(periods=4)

# Create the forecast
forecast = m.predict(future)

forecast.tail()
m.plot(forecast)
m.plot_components(forecast)
forecast[['yhat','yhat_upper','yhat_lower']] = forecast[['yhat','yhat_upper','yhat_lower']].apply(lambda x: inv_boxcox(x, lam))
forecast = forecast[['ds','yhat','yhat_lower','yhat_upper']]
forecast = forecast.set_index('ds')

forecast.tail()







