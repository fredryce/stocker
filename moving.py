
from finta import TA
import scipy as sp
from scipy.signal import argrelextrema
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import yfinance as yf
from collections import defaultdict


class Loss(object):
	def __init__(self,method, sup, res):
		self.method = method
		self.sup = sup
		self.res = res
	def __repr__(self):
		return f"{self.method} sup:{self.sup} res:{self.res}"


class Kline(object):
	def __init__(self):
		self.buy = [self.double_bottom, self.hammer_head]
		self.sell = []
	def double_bottom(self, row):
		pass
	def hammer_head(self, row):
		#when decreasing
		pass



class MA(object): #using different combinations
	def __init__(self,yahoo_tick):
		self.days = [5,10,20,50,80,120,180,200]
		self.fibdays = [8,21,34,55,89,144,233]
		self.madict = {} #key:{day:pddf} #everything in here is shown in the plot
		self.plotma = [] #ma lines to be plotted
		self.maxdaylength = max(max(self.days), max(self.fibdays))

		self.expected_num = 20
		self.shiftAmount = 7 #how many max to look back
		self.chosen_sup ={}
		self.chosen_res = {}

		self.half_hist = yahoo_tick.history(period=f"60d", interval="30m")
		self.hour_hist = yahoo_tick.history(period=f"60d", interval="1h")
		self.__hist = yahoo_tick.history(period=f"{self.maxdaylength*2}d", interval="1d")
		self.week_hist = yahoo_tick.history(period=f"{self.maxdaylength*2}d", interval="5d")
		self.month_hist = yahoo_tick.history(period=f"{self.maxdaylength*2}d", interval="1mo")


		self.get_MA(self.hist)



	def refresh(self, choice=None):
		if choice=="hour":
			self.hour_hist = yahoo_tick.history(period=f"60d", interval="1h")
		elif choice == "half":
			self.half_hist = yahoo_tick.history(period=f"60d", interval="30m")
		elif choice == "week":
			self.week_hist = yahoo_tick.history(period=f"{self.maxdaylength*2}d", interval="5d")
		elif choice == "month":
			self.month_hist = yahoo_tick.history(period=f"{self.maxdaylength*2}d", interval="1mo")
		elif choice == "day":
			self.__hist = yahoo_tick.history(period=f"{self.maxdaylength*2}d", interval="1d")
		else:
			self.half_hist = yahoo_tick.history(period=f"60d", interval="30m")
			self.hour_hist = yahoo_tick.history(period=f"60d", interval="1h")
			self.__hist = yahoo_tick.history(period=f"{self.maxdaylength*2}d", interval="1d")
			self.week_hist = yahoo_tick.history(period=f"{self.maxdaylength*2}d", interval="5d")
			self.month_hist = yahoo_tick.history(period=f"{self.maxdaylength*2}d", interval="1mo")


	@property
	def hist(self):
		return self.__hist

	@hist.setter
	def hist(self, value):
		self.__hist = value
		self.get_MA(value)


	def get_MA(self, hist):
		new_cols = [x.lower() for x in hist.columns]
		hist.columns = new_cols
		for day in self.days:
			self.madict[f"SMA_{day}"] = TA.SMA(hist, day)
			self.madict[f"SMA_{day}"].name = f"SMA_{day}"
		for day in self.days:
			self.madict[f"WMA_{day}"] = TA.WMA(hist, day)
		for day in self.days:
			self.madict[f"HMA_{day}"] = TA.HMA(hist, day)
		for day in self.days:
			self.madict[f"EMA_{day}"] = TA.EMA(hist, day)

		for day in self.fibdays:
			self.madict[f"SMA_{day}F"] = TA.SMA(hist, day)
		for day in self.fibdays:
			self.madict[f"WMA_{day}F"]  = TA.WMA(hist, day)
		for day in self.fibdays:
			self.madict[f"HMA_{day}F"]  = TA.HMA(hist, day)
		for day in self.fibdays:
			self.madict[f"EMA_{day}F"] = TA.EMA(hist, day)

		mavalue = self.madict[f"SMA_50"]


		sma50 = np.argwhere(np.isnan(mavalue.values)).flatten()
		curve_point = np.setxor1d(np.argwhere(np.diff(np.sign(self.madict[f"SMA_20"].values - mavalue.values))).flatten(), sma50)

		indexs = mavalue.iloc[curve_point]
		dff = self.madict[f"SMA_50"].to_frame('50dfvalue')
		dff["inter"] = np.nan
		dff.loc[indexs.index,"inter"] = mavalue.loc[indexs.index]

		self.overlap_sma2050 = dff["inter"]

		#print(dff)

		'''

		plt.plot(self.madict[f"SMA_50"].index, self.madict[f"SMA_50"] , "-o")
		plt.plot(self.madict[f"SMA_20"].index, self.madict[f"SMA_20"] , "-o")
		plt.plot(self.overlap_sma2050.index, self.overlap_sma2050, "o")
		plt.show()
		exit()
		'''



	def findATR(self, hist):
		result= TA.ATR(hist)
		return result.dropna()


	def findRSI(self, hist):
		result = TA.RSI(hist, 14)
		return result.dropna()

	def update_hists(self, row_data):
		#update half, hour, day, week, month
		pass

	def find_max_min(self, hist): #find all the max points and all the min points
		
		arr_size = len(hist.Bottom)
		expected_width = arr_size // self.expected_num // 2
		print('expected width of peaks: ', expected_width)
		maximaIdxs = sp.signal.find_peaks_cwt(hist.Bottom, np.linspace(2, expected_width, 10))
		minimaIdxs = sp.signal.find_peaks_cwt(-1*hist.Bottom, np.linspace(2, expected_width, 10))
		hist["critical"] = ""
		old_index = hist.index.name
		hist = hist.reset_index()
		hist.loc[minimaIdxs, "critical"] = "min"
		hist.loc[maximaIdxs, "critical"] = "max"
		hist = hist.set_index(old_index)
		hist = self.findSupandRes(hist)

		return hist

	def findSupandRes(self, hist):
		lossvalue = {}
		for method, series in self.madict.items():
			series.name = method

			old_index = hist.index
			if hist.index.name == "Datetime":
				hist.index = hist.index.date
			hist = pd.merge(hist, series,left_index=True, right_index=True)

			hist.index = old_index

			min_df = hist[hist["critical"] == "min"]
			min_df_shift = pd.DataFrame(index=min_df.index)

			for i in range(1, self.shiftAmount+1):
				min_df_shift[[f'{method}_{i}', f'Bottom_{i}']] = min_df[[method, "Bottom"]].shift(i)
				#min_df_shift[[f'{method}_{-i}', f'Bottom_{-i}']] = min_df[[method, "Bottom"]].shift(-i, fill_value=0)

	

			max_df = hist[hist["critical"] == "max"] 
			max_df_shift = pd.DataFrame(index=max_df.index)

			for i in range(1, self.shiftAmount+1):
				max_df_shift[[f'{method}_{i}', f'Bottom_{i}']] = max_df[[method, "Bottom"]].shift(i)
				#max_df_shift[[f'{method}_{-i}', f'Bottom_{-i}']] = max_df[[method, "Bottom"]].shift(-i, fill_value=0)

			

			sup_cond, res_cond = self.findcondition(hist, method, min_df_shift, max_df_shift)

			hist.loc[min_df.index, f'{method}valuemin'] = sup_cond
			hist.loc[max_df.index, f'{method}valuemax'] = res_cond

			



			sup_count =  hist.loc[min_df.index, f'{method}valuemin'].sum()
			res_count = hist.loc[max_df.index, f'{method}valuemax'].sum()


			self.chosen_sup[method] = sup_cond
			self.chosen_res[method] = res_cond


			lossvalue[method] = Loss(method, sup_count, res_count)

		print(lossvalue)
		
		key_min_sup = max(lossvalue, key=lambda k: lossvalue[k].sup)
		key_min_res = max(lossvalue, key=lambda k: lossvalue[k].res)

		supMethod = lossvalue[key_min_sup].method
		resMethod = lossvalue[key_min_res].method

		
		sup_cond = self.chosen_sup[supMethod]
		index_v = sup_cond[sup_cond==1].index
		plt.plot(sup_cond[sup_cond==1].index, hist.loc[index_v, "close"], "o", markersize=12, label="sup")
			
	
		res_cond = self.chosen_res[resMethod]
		index_v = res_cond[res_cond==1].index
		plt.plot(res_cond[res_cond==1].index, hist.loc[index_v, "close"], "o", markersize=12, label="res")
		

		print(f"{lossvalue[key_min_sup].method} sup:{lossvalue[key_min_sup].sup}")
		print(f"{lossvalue[key_min_res].method} res:{lossvalue[key_min_res].res}")

		if lossvalue[key_min_sup].sup != 0:
			self.plotma.append(lossvalue[key_min_sup].method)
		if lossvalue[key_min_res].res != 0:
			self.plotma.append(lossvalue[key_min_res].method)
			


		#self.plotma.append("HMA_80")
		
		return hist

	def findcondition(self, hist, method, min_df, max_df):
		col_names = min_df.columns
		min_df["all_met"] = min_df.apply(lambda row : self.filter_condition(row, "min", col_names), axis = 1)

		max_df["all_met"] = max_df.apply(lambda row : self.filter_condition(row, "max", col_names), axis = 1)

		
		sup_cond = (hist.loc[min_df.index, method] <= hist.loc[min_df.index, 'Bottom']) & \
					(hist.loc[min_df.index, method] >= hist.loc[min_df.index, 'low'])& \
					min_df["all_met"]
					#(hist.loc[min_df.index, method] >= hist.loc[min_df.index, 'Bottom'] - hist.loc[min_df.index, 'Bottom']*self.threshold) & \

		res_cond = (hist.loc[max_df.index, method] >= hist.loc[max_df.index, 'Bottom']) & \
					(hist.loc[max_df.index, method] <= hist.loc[max_df.index, 'high']) & \
					max_df["all_met"]
					#(hist.loc[max_df.index, method] <= hist.loc[max_df.index, 'Bottom'] + hist.loc[max_df.index, 'Bottom']*self.threshold) & \

		return sup_cond.astype(int), res_cond.astype(int)

	def filter_condition(self, row, target, col_names):
		
		results = []

		if target == "min":
			for i, value in enumerate(row):
				if i % 2 == 0:
					results.append(value <= row[col_names[i+1]])
		if target == "max":
			for i, value in enumerate(row):
				
				if i % 2 == 0:
					results.append(value >= row[col_names[i+1]])

		return np.all(results)



if __name__ == "__main__":
	smoothing = 3
	window = 10
	yahoo_tick = yf.Ticker("SENS")
	myMA = MA(yahoo_tick)
	ticks = ["SENS", "GIK", "NNDM", "SPY"]
	ema_list = [5]
	window_list = [5]

	results = myMA.screener(ticks, ema_list, window_list, plot=True, results=True)
	print(results)
	plt.show()

	#minmax = myMA.get_max_min(smoothing, window)
	#print(minmax)