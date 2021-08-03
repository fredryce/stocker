
import pandas as pd
from finta import TA
import mplfinance as mpf
from pandas_datareader import data as web
from matplotlib import pyplot as plt
import time
from matplotlib.animation import FuncAnimation
import yfinance as yf


# Finta likes lowercase
#data.columns = ["open", "high", "low", "close", "volume"]

# calc bol band
#bbands = TA.BBANDS(data, 30)

# cherry pick what to show on the chart
#bands_plot = pd.concat([bbands.BB_UPPER, bbands.BB_LOWER], axis=1)

#print(bands_plot)

#apd = mpf.make_addplot(bands_plot.head(300))



'''

test = hist.tail(400)
fig= plt.figure()


def animate(i):
    global fig

    #cansee = test.iloc[:i+1,:]
    cansee = yahoo_tick.history(period="1d", interval="5m")
    print(cansee)
    #cansee.index = cansee["index"]
    plt.clf()

    fig, axs = mpf.plot(cansee, fig, type='candle', style='yahoo',
    title=f'{ticker} (30)',
    ylabel='Price (USD)',
    ylabel_lower='Volume',
    volume=True,
    figscale=1.5,
    animate=True
    )

    time.sleep(0.01)




#plt.tight_layout()
plt.show()
'''













