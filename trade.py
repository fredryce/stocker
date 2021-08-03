class Trades:
	starting = 1000
	instances = []
	stock_count = 0

	def __init__(self, stock_ticker, wb, id_value, cost, buy=0, sell=0, profitsl=None, realtime=False, islimit=False, finished=False):
		self.wb = wb
		self.stock_ticker=stock_ticker
		
		self.profitsl = profitsl
		self.id_value = id_value
		self.cost = cost
		self.buy = buy
		self.sell = sell
		self.realtime = realtime
		self.islimit =islimit #checking if its limit sell
		self.finished = finished #if positions has been sold

		if realtime:
			acc_data =self.wb.get_account()
			Trades.starting = float(acc_data["accountMembers"][1]["value"])
			Trades.stock_count = self.find_positions(acc_data["positions"])
			self.sell_stock()
			self.buy_stock()
		else:
			self.process_hist_buysell()

		Trades.instances.append(self)

	def process_hist_buysell(self):
		if Trades.starting >= (self.cost*self.buy):#make sure enough money to buy
			Trades.stock_count += self.buy #buy stock amount
			Trades.starting -= (self.cost*self.buy)
		else:
			Trades.stock_count += (Trades.starting / self.cost)
			Trades.starting = 0


		if Trades.stock_count >= self.sell: #make sure enough stock to sell
			Trades.stock_count -= self.sell
			Trades.starting += (self.cost*self.sell)
		else:
			Trades.starting += (Trades.stock_count * self.cost)
			Trades.stock_count = 0


	def sell_stock(self): #given amount to buy and sell update the self variables
		#self.wb.get_trade_token('199694')
		if (not Trades.stock_count) or (not self.sell):
			print("No stocks to sell")
			return

		self.wb.get_trade_token('199694')
		if Trades.stock_count >= self.sell: #make sure enough stock to sell
			
			sucess_dict = self.wb.place_order(stock=self.stock_ticker, tId=None, action='SELL', orderType='MKT', enforce='DAY', quant=self.sell)


			print(f"Im selling ${self.cost*self.sell} of {self.stock_ticker} at {self.cost} {self.sell} stocks Enough")
			print(f"Finshed selling {self.stock_ticker}... {sucess_dict}")

			Trades.stock_count -= self.sell
			Trades.starting += (self.cost*self.sell)
		else:
			
			sucess_dict = self.wb.place_order(stock=self.stock_ticker, tId=None, action='SELL', orderType='MKT', enforce='DAY', quant=Trades.stock_count)

			print(f"Im selling ${self.cost*Trades.stock_count} of {self.stock_ticker} at {self.cost} {Trades.stock_count} stocks not Enough")
			print(f"Finshed selling {self.stock_ticker}... {sucess_dict}")
			Trades.starting += (Trades.stock_count * self.cost)
			Trades.stock_count = 0



	def buy_stock(self):

		if (not Trades.starting) or (not self.buy):
			print("No money to buy")
			return
		self.wb.get_trade_token('199694')
		if Trades.starting >= (self.cost*self.buy):#make sure enough money to buy
			
			sucess_dict = self.wb.place_order(stock=self.stock_ticker, tId=None, action='BUY', orderType='MKT', enforce='DAY', quant=self.buy)

			print(f"Im buying ${self.cost*self.buy} of {self.stock_ticker} at {self.cost} {self.buy} money Enough")
			print(f"Finshed Buying {self.stock_ticker}...{sucess_dict}")
			Trades.stock_count += self.buy #buy stock amount
			Trades.starting -= (self.cost*self.buy)

		else:
			
			sucess_dict = self.wb.place_order(stock=self.stock_ticker, tId=None, action='BUY', orderType='MKT', enforce='DAY', quant=Trades.starting/self.cost)

			print(f"Im buying ${Trades.starting} of {self.stock_ticker} at {self.cost} {Trades.starting/self.cost} money not Enough")
			print(f"Finshed Buying {self.stock_ticker}... {sucess_dict}")
			Trades.stock_count += (Trades.starting / self.cost)
			Trades.starting = 0

	def find_positions(self, position_data):
		for position in position_data:
			if position["ticker"]["symbol"] ==self.stock_ticker:
				return float(position["position"])
		return 0


class ProfitSL(object):
	def __init__(self, profit=None, sl=None):
		self.profit=profit
		self.sl = sl

	def __str__(self):
		return f"profit: {self.profit} stopl: {self.sl}"



