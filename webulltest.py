import pymongo
from finviz.screener import Screener
#client = pymongo.MongoClient("mongodb://xin:macbook@datalake0-yamct.a.query.mongodb.net/myFirstDatabase?ssl=true&authSource=admin")
#db = client.test
# streamlit run webulltest.py
'''
Issues:
	when asking for data, the timestamps is not consistant, need another data source or get rsi values directly

'''

import json
import trendln
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import sched
import time
import traceback
import yfinance as yf
from matplotlib.animation import FuncAnimation
import mplfinance as mpf
from mplfinance._utils import IntegerIndexDateTimeFormatter
from mplfinance._arg_validators import _check_and_prepare_data

from findstock import *

from finta import TA
import matplotlib.dates  as mdates
import logging
import os, sys, glob
import requests
from util import *
import streamlit as st

import warnings
warnings.filterwarnings("ignore")

#should be scared of fridays and during lunch hours price tends to go down




class Plotdata(object):
	def __init__(self, ticker, wb, period="1", timeframe="1", count =None, backtest=False, trunc=True, extendedhour=0, gui=True, stream=False, local_csv=None):
		self.ticker = ticker
		self.wb = wb
		self.period = period
		self.stream=stream
		self.timeframe = timeframe
		self.strat_object = MA2050(self.ticker, wb)
		self.strat_object.extendTrading = extendedhour
		self.strat_object.backtest = backtest
		self.backtest = backtest
		self.local_csv = local_csv
		self.history_df = self.strat_object.preprocess(period,timeframe, count, local_csv=local_csv) #get the df of the default period, if the last item of data pass the last point of history then we need to generate a new set
		self.offset=14
		self.trunc = trunc

		if gui:
			self.fig = plt.figure()
			self.window_size = 100
			self.run()

	def save_csv(self):
		self.strat_object.history_df.to_csv(f"{self.ticker}_result.csv")
		print("saving....")
		plt.savefig(f"{self.ticker}_result.png")
		print("saved...")


	def animate(self, i):

		if not self.backtest:
			self.history_df = self.strat_object.process_data(self.timeframe) #contain the atr, rsi, buy and sell


		final_data = self.history_df.iloc[self.offset:,:].copy()

		if self.backtest:
			final_data = final_data.iloc[:i+1,:]
		
		if self.trunc:
			if len(final_data.index) >= self.window_size:
				final_data= final_data.iloc[-self.window_size:, :]



		copy_df = final_data.copy()

		final_data['buy'] = final_data["buy"].replace(0, np.nan)
		final_data['sell'] = final_data["sell"].replace(0, np.nan)

		buy_columns  = final_data[final_data["buy"].isna()]
		sell_columns = final_data[final_data["sell"].isna()]
		

	
		copy_df.loc[buy_columns.index, "low"] = np.nan
		copy_df.loc[sell_columns.index, "high"] = np.nan

		


		apd = [mpf.make_addplot(final_data["rsi"], panel="lower")]

		#apd.append(mpf.make_addplot(self.strat_object.ma.madict["SMA_20"].loc[final_data.index.date]))
		#apd.append(mpf.make_addplot(self.strat_object.ma.madict["SMA_50"].loc[final_data.index.date]))

		

		final_data["last_price"] = np.nan
		final_data.loc[final_data.index[-1], "last_price"]  = final_data.iloc[-1,:].close
		



		apd.append(mpf.make_addplot(final_data["last_price"] ,scatter=True,markersize=len(str(final_data.iloc[-1,:].last_price)) * 200,marker=f'${final_data.iloc[-1,:].last_price}$'))


		if not copy_df["low"].isnull().all():
			apd.append(mpf.make_addplot(copy_df["low"] -0.05 ,scatter=True,markersize=100,marker='^', color='g'))



		if not copy_df["high"].isnull().all():
			apd.append(mpf.make_addplot(copy_df["high"]+ 0.05 ,scatter=True,markersize=100,marker='v', color='r'))

		
		SMA_cross_over =self.strat_object.ma.overlap_sma2050

		if (not SMA_cross_over.isnull().all()) and (not SMA_cross_over.empty):
			SMA_cross_over_df = SMA_cross_over.loc[final_data.index.date]   
			if not SMA_cross_over_df.isnull().all():
				apd.append(mpf.make_addplot(SMA_cross_over_df, scatter=True, markersize=200))

		#mpf.make_addplot(copy_df.high ,scatter=True,markersize=100,marker='v')]#, mpf.make_addplot(MA50.dropna())]  


		plt.clf()

		
		self.fig, axs = mpf.plot(final_data, self.fig, type='candle', style='yahoo',
		title=f'{self.ticker} (30)',
		ylabel='Price (USD)',
		ylabel_lower='Volume',
		volume=True,
		figscale=2,
		returnfig=True,
		addplot=apd,
		block=False
		)

	
		
	  
	
		time.sleep(0.0001)

	def run(self):
		try:
			if not self.stream:
				print("Running local visualization.......")
				ani = FuncAnimation(self.fig, self.animate, interval=1000)
				plt.show()
			else:
				print("Running browser visualization.......")
				the_plot = st.pyplot(plt)
				counter = 0
				if self.local_csv:
					exit()
				while True:
					self.animate(counter)
					the_plot.pyplot(plt)
					counter+=1

		except KeyboardInterrupt:
			print("Bye")
			self.save_csv()
			exit()

	



if __name__ == "__main__":

	wb = login()
	dt = datetime.utcnow().timestamp()
	dir_name = os.path.join('./loggings', str(dt))
	os.mkdir(dir_name)

	print("making dir ", dir_name)
	logging.basicConfig(filename=os.path.join(dir_name, 'output.log'), filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)



	period = '1' #days do you want to use to calculate support/resistance (e.g. 1,5,30) 
	symbol = "SENS"
	timeframe = '1' #timeframe in minutes to trade on (e.g. 1,5,15,60



	pdclass = Plotdata(symbol,wb, period=period, timeframe=timeframe, backtest=False, trunc=False, extendedhour=1, gui=True, stream=False, local_csv=None) #need
	
	#test_all_stocks(wb)


