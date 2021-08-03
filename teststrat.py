from trade import *
from datetime import datetime,timedelta
import yfinance as yf
from moving import *

class PatternsandCandles(object): #this is use to determine enter and exit based on patterns and candles

	def __init__(self):
		self.buy = ["double bottom", "inverse head and shoulder"]
		self.sell = ["double top", "head and shoulder"]
		#it should have a probability for each of the me
		#take in the current data, 
		#try multiple groups, increament the window size on one end
		#1min -29
		#30 - 59
		#1h - 24
		#1day-6day
		#
		self.smoothing = 5
		self.window_range = 5
	def double_bottom(self):
		pass
	#this needs to have acesss to ma object and df
	def swing_low_price(self, row_value, number_up=2, period=60):

		temp_focus = self.history_df.index.get_loc(row_value.name)

		if temp_focus - period <0:
			temp_focus=60

		lookback = self.history_df.iloc[temp_focus-period:temp_focus+1, :]

		max_min = self.get_max_min(lookback, target="min")

		if max_min.empty:
			return row_value.low, row_value.name

		return max_min.iloc[-1], lookback.index[max_min.index[-1]]

	def IHS(self):
		pass



	def get_max_min(self, hist, smoothing=None, window_range=None, target="both"):
		if not smoothing:
			smoothing =self.smoothing
		if not window_range:
			window_range = self.window_range

		index_name = hist.index.name
		smooth_prices = hist['close'].rolling(window=smoothing).mean().dropna()
		local_max = argrelextrema(smooth_prices.values, np.greater)[0]
		local_min = argrelextrema(smooth_prices.values, np.less)[0]
		price_local_max_dt = []
		for i in local_max:
			if (i>window_range) and (i<len(hist)-window_range):
				price_local_max_dt.append(hist.iloc[i-window_range:i+window_range]['close'].idxmax())
		price_local_min_dt = []
		for i in local_min:
			if (i>window_range) and (i<len(hist)-window_range):
				price_local_min_dt.append(hist.iloc[i-window_range:i+window_range]['close'].idxmin())  

		if target == "max":
			max_min = pd.DataFrame(hist.loc[price_local_max_dt])
		elif target == "min":
			max_min = pd.DataFrame(hist.loc[price_local_min_dt])
		else:
			max_min = pd.concat([pd.DataFrame(hist.loc[price_local_max_dt]), pd.DataFrame(hist.loc[price_local_min_dt])]).sort_index()
			
		max_min.index.name = 'date'

		max_min = max_min.reset_index()
		max_min = max_min[~max_min.date.duplicated()]
		p = hist.reset_index() 
		max_min['day_num'] = p[p[index_name].isin(max_min.date)].index.values
		max_min = max_min.set_index('day_num')['close']

		#print(max_min)
		#hist.reset_index()["close"].plot()
		#plt.plot(max_min.index, max_min.values, "o")
		#plt.show()

		return max_min


	def find_patterns(self, max_min):  
		patterns = defaultdict(list)
		letter_locations = {}
		
		# Window range is 5 units
		for i in range(5, len(max_min)):  
			window = max_min.iloc[i-5:i]
			
			# Pattern must play out in less than n units
			if window.index[-1] - window.index[0] > 100:      
				continue   
				
			a, b, c, d, e = window.iloc[0:5]
					
			# IHS
			if a<b and c<a and c<e and c<d and e<d and abs(b-d)<=np.mean([b,d])*0.02 and abs(a-e)<=np.mean([a,e])*0.000001:
				   patterns['IHS'].append((window.index[0], window.index[-1]))
				   letter_locations[(window.index[0], window.index[-1])] = [a,b,c,d,e]
		#print(letter_locations)
		return patterns, letter_locations


	def plot_minmax_patterns(self, max_min, patterns, stock, window, ema, letter_locations):
	
		incr = str((self.hist.index[1] - self.hist.index[0]).seconds/60)
		
		if len(patterns) == 0:
			pass
		else:
			num_pat = len([x for x in patterns.items()][0][1])
			f, axes = plt.subplots(1, 2, figsize=(16, 5))
			axes = axes.flatten()
			prices_ = self.hist.reset_index()['close']
			axes[0].plot(prices_)
			axes[0].scatter(max_min.index, max_min, s=100, alpha=.3, color='orange')
			axes[1].plot(prices_)


			for name, end_day_nums in patterns.items():
				for i, tup in enumerate(end_day_nums):
					sd = tup[0]
					ed = tup[1]
					axes[1].scatter(max_min.loc[sd:ed].index,
								  max_min.loc[sd:ed].values,
								  s=200, alpha=.3)
					plt.yticks([])
			plt.tight_layout()
			plt.title('{}: {}: EMA {}, Window {} ({} patterns)'.format(stock, incr, ema, window, num_pat))

	def get_results(self, max_min, pat, stock, ema_, window_):
		
		incr = str((self.hist.index[1] - self.hist.index[0]).seconds/60)
		
		#fw_list = [1, 12, 24, 36] 
		fw_list = [1, 2, 3]
		results = []
		if len(pat.items()) > 0:
			end_dates = [v for k, v in pat.items()][0]      
			for date in end_dates:  
				param_res = {'stock': stock,
							 'increment': incr,
							 'ema': ema_,
							 'window': window_, 
							 'date': date}
				for x in fw_list:
					returns = (self.hist['close'].pct_change(x).shift(-x).reset_index(drop=True).dropna())
					try:
						param_res['fw_ret_{}'.format(x)] = returns.loc[date[1]]   
					except Exception as e:
						param_res['fw_ret_{}'.format(x)] = e
				results.append(param_res)  
		else:
			param_res = {'stock': stock,
						 'increment': incr,
						 'ema': ema_,
						 'window': window_,
						 'date': None}
			for x in fw_list:
				param_res['fw_ret_{}'.format(x)] = None   
			results.append(param_res)
		return pd.DataFrame(results)

	def screener(self, stock_data, ema_list, window_list, plot, results=False):
			
		all_results = pd.DataFrame()
		
		for stock in stock_data:
			#prices = stock_data[stock]
			yahoo_tick = yf.Ticker(stock)
			self.hist = yahoo_tick.history(period=f"60d", interval="30m")
			
			for ema_ in ema_list:
				for window_ in window_list: 
					max_min = self.get_max_min(smoothing=ema_, window_range=window_)
					pat,location = self.find_patterns(max_min)
					
					if plot == True:
						self.plot_minmax_patterns(max_min, pat, stock, window_, ema_, location)
						
					if results == True:
						all_results = pd.concat([all_results, self.get_results(max_min, pat, stock, ema_, window_)], axis=0)
					
		if results == True:
			return all_results.reset_index(drop=True)

###########################################################
###########################################################
###########################################################
###########################################################

class Strat(PatternsandCandles): #main object for main functions to apply sell and buy #should determine the peroid for computing sell and buy
	def __init__(self, ticker, wb=None, extendTrading=0, backtest=False):
		super(Strat, self).__init__()
		self.ticker=ticker
		self.starting_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
		self.ending_time = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
		#self.stock_count = 0 #when buy increase this
		self.wb = wb
		self.backtest = backtest
		self.yahoo_tick = yf.Ticker(self.ticker)
		self.ma = MA(self.yahoo_tick)
		self.extendTrading=extendTrading
		self.timeouttick = 0
		self.timeoutmax = 200

		self.required_period = None #this df is required for each strat as the dafult dict to perform analysis on. if its none then the stock can be used on real time data
		self.required_timestamp = None

	def process_df(self, hist):
		#modi to hist
		hist["buy"] = 0
		hist["sell"] = 0
		hist["Bottom"] = np.where(hist["close"] >= hist["open"], hist["open"], hist["close"])
		hist["Top"] = np.where(hist["close"] >= hist["open"], hist["close"], hist["open"])

		return hist

	def preprocess(self, period, timestamp, count):
		#called from the plot main function to setup var before running
		print(f"Running preprocess....")
		if not self.required_period:
			self.required_period = period
		if not self.required_timestamp:
			self.required_timestamp = timestamp

		if self.backtest:
			self.history_df = self.yahoo_tick.history(period=f"{self.required_period}d", interval=f"{self.required_timestamp}m")
			self.history_df.columns = [x.lower() for x in self.history_df.columns]
		else:
			self.history_df = self.wb.get_bars(stock=self.ticker.upper(), interval='m'+self.required_timestamp, count=int((390*int(self.required_period))/int(self.required_timestamp)), extendTrading=self.extendTrading)


		self.history_df = self.process_df(self.history_df)

		start = Trades.starting

		try:
			self.history_df = self.history_df.apply(lambda row : self.buy_sell(row, realtime=False), axis = 1)

			percent_hold = ((((start / self.history_df["open"].iloc[0]) * self.history_df["close"].iloc[-1])-start) / start)*100 #percent gain if just hold

			percent = (((Trades.starting + (Trades.stock_count * self.history_df.iloc[-1,:].close))-start)/start)*100

			self.df_result_list = [self.ticker, start, self.determine_total_worth(self.history_df.iloc[-1,:].close), Trades.starting, Trades.stock_count * self.history_df.iloc[-1,:].close, percent]
			print(f"RESULT - ${start} - ${Trades.starting + (Trades.stock_count * self.history_df.iloc[-1,:].close)}. liquid:${Trades.starting} stocks:${Trades.stock_count * self.history_df.iloc[-1,:].close} {percent}% change. Normal gain {percent_hold}%")

		except Exception as e:
			self.df_result_list = [self.ticker, f"{traceback.format_exc()}", None, None, None, None]
			print(f"{self.ticker} preprocess failed... {e}")
			logging.info(f"{self.ticker} failed... {e} {traceback.format_exc()}")



		

		return self.history_df


	def process_data(self, timestamp): #read 1 min data and return the most up to date df
		#this is called every iteration
		#multiple singals at one timestamp
		#problem, when data is 

		while True:
			try:
				data = self.wb.get_bars(stock=self.ticker.upper(), interval='m'+self.required_timestamp, count=1, extendTrading=self.extendTrading)
				break
			except Exception as e:
				print(f"{self.ticker} failed getting lastest bar retrying... {e}")
				if self.timeouttick > self.timeoutmax:
					raise Exception(f"{self.ticker} max tries exceed...")
				self.timeouttick += 1

		data = self.process_df(data)


		last_row_data = data.tail(1).squeeze()

		
		current_row = self.buy_sell(last_row_data, realtime=True)
		self.history_df.loc[current_row.name] = current_row
		

		self.ma.update_hists(data)
		return self.history_df

	def buy_sell(self, row_value): #at 1 min interval
		pass

	def determine_exit(self, row_value):#look for the trades stoploss
		row_value_temp = row_value.copy()
		signals = {'buy':0, 'sell':0}
		#for trade in self.trades:
		for trade in Trades.instances:
			if trade.profitsl and trade.buy and (not trade.finished):
				if row_value.close >= trade.profitsl.profit or row_value.close <= trade.profitsl.sl:
						row_value_temp["buy"] = signals['buy']
						row_value_temp["sell"]= row_value.close * trade.buy
						trade.finished=True

						print(f"HIT for trade {trade.id_value} hit profit:{trade.profitsl.profit} stopl:{trade.profitsl.sl} bought: {trade.buy} shares: {row_value.open} profit:{trade.cost_sold  - trade.cost}")

						#trade.profit_total += (trade.cost_sold  - trade.cost)

						return row_value

		return row_value


	def determine_total_worth(self, current_stock_price, realtime=False):

		if realtime:
			acc_data =self.wb.get_account()

			try:
				net_liquid = float(acc_data["netLiquidation"])
				return net_liquid
			except KeyError: #this needs to be fixed when running multiprocessing updated globally to all the raise an error and update all other processes wb account object
				print(f"{self.ticker} No net liquid retrying the log... {acc_data}")
				from util import login
				self.wb = login(self.wb)
				acc_data =self.wb.get_account() #getting account id might still cause issue
				logging.info(f"{acc_data}")
				net_liquid = float(acc_data["netLiquidation"])
				print("retrying success...")
				return net_liquid

		return (Trades.stock_count * current_stock_price) + Trades.starting

	def vis(self,ticker,map_df):
		#vis of sell and buy points
		#df_expand = pd.json_normalize(map_df['signals'])
		sell_rows = map_df[map_df["sell"] != 0]
		buy_rows = map_df[map_df["buy"] != 0] 
		plt.plot(map_df.index, map_df.Bottom, "-")
		plt.plot(buy_rows.index, buy_rows.close, "go", label="BUY")
		plt.plot(sell_rows.index, sell_rows.close, "ro", label="SELL")
		for item in self.ma.plotma:
			#print("plotting.... ", item)
			plt.plot(self.ma.madict[item].index, self.ma.madict[item], "-", label=item)

		plt.legend()
		plt.show()

	def testHistory(self, period="60d", interval='30m', process_function=None): #process function is passed in after applying to the dataframe
		start = self.starting_price_current
		hist = self.yahoo_tick.history(period=period, interval=interval)
		hist.columns = [x.lower() for x in hist.columns]
		hist["Bottom"] = np.where(hist["close"] >= hist["open"], hist["open"], hist["close"])
		if process_function: hist = process_function(hist)
		dict_list = hist.apply(lambda row : self.buy_sell(row), axis = 1)
		percent = (((self.starting_price_current + (self.stock_count * hist.iloc[-1,:].close))-start)/start)*100
		print(f"${start} - ${self.starting_price_current + (self.stock_count * hist.iloc[-1,:].close)}. liquid:${self.starting_price_current} stocks:${self.stock_count * hist.iloc[-1,:].close} {percent}% change")
		self.vis(hist, dict_list)


###########################################################
###########################################################
###########################################################
###########################################################




class MA2050(Strat):
	#https://www.youtube.com/watch?v=hTDVTH8umR8&t=1573s&ab_channel=TheTradingChannel
	#find crossover of 50 and 20day moving average, use ATR to determine exit strat
	def __init__(self, ticker,wb):
		super(MA2050, self).__init__(ticker=None)
		self.buy_percentage  = 0.01
		self.sell_percentage = 0.01

		self.rsi_low = 35
		self.rsi_high = 65

		self.triggered = 0 #if crossed low or high set triggered to true

		#blow are the parameters for determine divergence
		self.prev_high = None
		self.current_high = None

		self.prev_low = None
		self.current_low = None

		self.prev_rsi = None

	def process_df(self, hist):
		hist = super().process_df(hist)
		hist["rsi"] = self.ma.findRSI(hist)
		hist["atr"] = self.ma.findATR(hist)
		return hist

	def buy_sell(self, realtime=False):
		#strat based on 50ema, 20 ema, ATR and RSI

		row_value = self.history_df.index.iloc[-1]
		
		indicator_values = {"rsi":0, 'atr':0}
		signals = {'buy':0, 'sell':0}


		df_tocurrent = self.history_df.copy()


		if not df_tocurrent.empty:
			try:
				indicator_values["rsi"] = self.ma.findRSI(df_tocurrent).loc[row_value.name]


				self.prev_rsi = indicator_values["rsi"]
				

				
			except Exception as e:
				indicator_values["rsi"] = self.prev_rsi
				

			indicator_values["atr"] = self.ma.findATR(df_tocurrent).loc[row_value.name]





		RSI_value = indicator_values["rsi"]

		#print("RSI: ", RSI_value)

		ATR_value = indicator_values["atr"]

		

		#get stop loss value, determine atr, retrace footsteps
		low_price, index_time = self.swing_low_price(row_value) #checking stoplossses

		if low_price > row_value.close:
			low_price = row_value.close
			index_time = row_value.name

		stop_loss_price = low_price  - ATR_value
		price_dff = row_value.close - stop_loss_price
		profit_take = row_value.close + 1.4*price_dff


		#print(self.history_df)

		

		if (RSI_value) and (ATR_value):
			#print(f"ATR: {ATR_value} targetprofit: {profit_take} stoploss: {stop_loss_price} low: {low_price} low_time: {index_time} current: {row_value.close}")
			if RSI_value <= self.rsi_low:
				if not self.triggered: #when triggered is still 0 
					#first time entering in
					self.current_low = row_value.name #first time enter
					print(f'{self.ticker} ENTERED rsi of {self.rsi_low} with {RSI_value} current low is {self.current_low}')
				else:
					if self.current_low:
						if RSI_value < self.history_df.loc[self.current_low, "rsi"]:
							print(f'{self.ticker} UPDATE {RSI_value} < {self.history_df.loc[self.current_low, "rsi"]} at {self.current_low}')
							self.current_low = row_value.name


				self.triggered=self.rsi_low

				


			elif RSI_value >= self.rsi_high:
				if not self.triggered:
					self.current_high = row_value.name
					print(f'{self.ticker} ENTERED rsi of {self.rsi_high} with {RSI_value} current high is {self.current_high}')
				else:
					if self.current_high:
						if RSI_value > self.history_df.loc[self.current_high, "rsi"]:
							print(f'{self.ticker} UPDATE {RSI_value} > {self.history_df.loc[self.current_high, "rsi"]} at {self.current_high}')
							self.current_high = row_value.name




				self.triggered = self.rsi_high

				


			if self.triggered ==self.rsi_low and RSI_value >= self.rsi_low: #exiting the trade when it crosses back over. determine slope and change prev, set current

				print(f'{self.ticker} EXIT {RSI_value} > {self.rsi_low}. current low at {self.current_low}, prevlow at {self.prev_low}')


				if self.prev_low and self.current_low:
					#can compare
					rsi_slope = self.history_df.loc[self.current_low, "rsi"] - self.history_df.loc[self.prev_low, "rsi"]
					price_slope = self.history_df.loc[self.current_low, "close"] - self.history_df.loc[self.prev_low, "close"]
					if ((rsi_slope>0) and (price_slope<0)):
						signals["buy"] = self.determine_total_worth(row_value.close, realtime) * self.buy_percentage
						print(f'{self.current_low} BUYING based on prev low at time {self.prev_low}. price_old: ${self.history_df.loc[self.prev_low, "close"]}, price_new: ${self.history_df.loc[self.current_low, "close"]}, rsi_old: {self.history_df.loc[self.prev_low, "rsi"]} rsi_new: {self.history_df.loc[self.current_low, "rsi"]}')

				
				self.prev_low = self.current_low
				self.current_low = None
				self.triggered=0

			if self.triggered == self.rsi_high and RSI_value <= self.rsi_high:

				print(f'{self.ticker} EXIT {RSI_value} < {self.rsi_high}. current high at {self.current_high}, prevlow at {self.prev_high}')


				if self.prev_high and self.current_high:
					#can compare
					rsi_slope = self.history_df.loc[self.current_high, "rsi"] - self.history_df.loc[self.prev_high, "rsi"]
					price_slope = self.history_df.loc[self.current_high, "close"] - self.history_df.loc[self.prev_high, "close"]
					if ((rsi_slope<0) and (price_slope>0)):
						signals["sell"] = self.determine_total_worth(row_value.close, realtime) * self.sell_percentage

						print(f'{self.current_high} SELLING based on prev high at time {self.prev_high}. price_old: ${self.history_df.loc[self.prev_high, "close"]}, price_new: ${self.history_df.loc[self.current_high, "close"]}, rsi_old: {self.history_df.loc[self.prev_high, "rsi"]} rsi_new: {self.history_df.loc[self.current_high, "rsi"]}')

				
				self.prev_high = self.current_high
				self.current_high = None
				self.triggered=0
		

		row_value["buy"] = signals['buy']
		row_value["sell"]= signals['sell']
		row_value["atr"] = indicator_values["atr"]
		row_value["rsi"] = indicator_values['rsi']


		return row_value

