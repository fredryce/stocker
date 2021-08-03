from findstock import *
import os, sys, glob

class MA2050(Strat):
	#https://www.youtube.com/watch?v=hTDVTH8umR8&t=1573s&ab_channel=TheTradingChannel
	#find crossover of 50 and 20day moving average, use ATR to determine exit strat
	def __init__(self, ticker,wb):
		super(MA2050, self).__init__(ticker,wb)
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


	def determine_divergence(self, target):
		pass

	def buy_sell(self, row_value, realtime=False):
		#strat based on 50ema, 20 ema, ATR and RSI

		already_action = False
		indicator_values = {"rsi":0, 'atr':0}
		signals = {'buy':0, 'sell':0}


		#check if its in df, if its not
		try:
			index_current = self.history_df.index.get_loc(row_value.name) #this is called before appending to the end of df so need to consider
			tail = self.history_df.tail(5)
			#print(f"{row_value.name} already in df check...index:{index_current} {self.history_df.iloc[index_current].buy} {self.history_df.iloc[index_current].sell} {tail[['open', 'close', 'buy', 'sell']]}")

			#print(f"{self.history_df.loc[row_value.name]}")

			if self.history_df.iloc[index_current].buy or self.history_df.iloc[index_current].sell:
				#print(f"already set buy {self.history_df.iloc[index_current].buy} {self.history_df.iloc[index_current].sell}")
				already_action = True


			else: #no buy or sell order from this timestamp
				self.history_df.loc[row_value.name] = row_value
				index_current = self.history_df.index.get_loc(row_value.name) 


		except KeyError:
			tail = self.history_df.tail(5)
			
			self.history_df.loc[row_value.name] = row_value
			index_current = self.history_df.index.get_loc(row_value.name) 
			#print(f"{row_value.name} not in df... {self.history_df.iloc[index_current].buy} {self.history_df.iloc[index_current].sell} {tail[['open', 'close', 'buy', 'sell']]}")



		df_tocurrent = self.history_df.iloc[index_current-20:index_current+1,:] #14day window based on existing hist

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

		

		if (RSI_value) and (ATR_value):
			#print(f"ATR: {ATR_value} targetprofit: {profit_take} stoploss: {stop_loss_price} low: {low_price} low_time: {index_time} current: {row_value.close}")
			if RSI_value <= self.rsi_low:
				self.triggered=self.rsi_low

				self.current_low = row_value.name #first time enter

				if not self.prev_low: # first time
					pass


			elif RSI_value >= self.rsi_high:
				self.triggered = self.rsi_high

				self.current_high = row_value.name



			if self.triggered ==self.rsi_low and RSI_value >= self.rsi_low:
				signals["buy"] = self.determine_total_worth(row_value.close, realtime) * self.buy_percentage
				self.triggered=0

			if self.triggered == self.rsi_high and RSI_value <= self.rsi_high:
				signals["sell"] = self.determine_total_worth(row_value.close, realtime) * self.sell_percentage
				self.triggered=0
		

		
		if already_action: #some acti
			signals['buy'] = self.history_df.iloc[index_current].buy
			signals['sell'] = self.history_df.iloc[index_current].sell



		row_value["buy"] = signals['buy']
		row_value["sell"]= signals['sell']
		row_value["atr"] = indicator_values["atr"]
		row_value["rsi"] = indicator_values['rsi']


		#row_value = self.determine_exit(row_value) #pass in thr row return a row


		if realtime and (not already_action):
			#no stop loss for long term trade
			#trade_v = Trade(row_value.name, cost=row_value.buy, buy= row_value.buy/row_value.open, sell = row_value.sell/row_value.open, profitsl=ProfitSL(profit=profit_take, sl = stop_loss_price))
			#amount_buy_sell = (row_value.buy or row_value.sell) / row_value.close
			profitsl = ProfitSL(profit=profit_take, sl = stop_loss_price)
			if row_value.buy or row_value.sell:
				Trades(self.ticker, self.wb, row_value.name, cost = row_value.close, buy=row_value.buy/row_value.close, sell=row_value.sell/row_value.close, profitsl=profitsl)
		
		else: #this is for preprocessing where its not enter market yet just for back testing
			if Trades.starting >= row_value.buy:#make sure enough money to buy
				Trades.stock_count += (row_value.buy/row_value.close) #buy stock amount
				Trades.starting -= row_value.buy

			if Trades.stock_count >= (row_value.sell/row_value.close): #make sure enough stock to sell
				Trades.stock_count -= (row_value.sell/row_value.close)
				Trades.starting += row_value.sell

		if not realtime:
			print(f"triggered value {self.triggered} {realtime}")
			print(f"{row_value.name} price:{row_value.close} BUY:{row_value.buy} SELL:{row_value.sell} ATR:{row_value.atr} RSI:{row_value.rsi}")
			print(f"Money:{Trades.starting} Count:{Trades.stock_count} total:{self.determine_total_worth(row_value.close)}")
		return row_value