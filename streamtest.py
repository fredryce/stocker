'''
import yfinance as yf
import streamlit as st

st.write("""
# Simple Stock Price App
Shown are the stock **closing price** and ***volume*** of Google!
""")

# https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
#define the ticker symbol
tickerSymbol = 'GOOGL'
#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
#get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')
# Open	High	Low	Close	Volume	Dividends	Stock Splits

st.write("""
## Closing Price
""")
st.line_chart(tickerDf.Close)
st.write("""
## Volume Price
""")
st.line_chart(tickerDf.Volume)
'''

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time

import mplfinance as mpf


from util import *
wb=login()
fig = plt.figure()
the_plot = st.pyplot(plt)

while True:

	history_df = wb.get_bars(stock="BTCUSD", interval='m1', count=100, extendTrading=1)

	plt.clf()

			
	fig, axs = mpf.plot(history_df, fig, type='candle', style='yahoo',
	ylabel='Price (USD)',
	ylabel_lower='Volume',
	volume=True,
	figscale=2,
	returnfig=True,
	block=False
	)
	the_plot.pyplot(plt)

	time.sleep(0.1)