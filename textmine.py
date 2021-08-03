import spacy
import sqlite3
import pandas as pd
import json
import robin_stocks as r
import yfinance as yf
import re, string
from os import makedirs, getcwd, path
import threading
from datetime import datetime, timedelta
#for the user, associate the user with the most recent stock he/she disccussed about
#we can use the basic youtube trading algo for long term invtestment
#use kelly formula, based the percentage on the faith of the discord chat

#https://www.youtube.com/watch?v=Hi-zhLgP_TQ&ab_channel=%E5%BC%82%E7%95%8C%E7%95%AA%E8%8C%84%E8%AF%B4%E7%BE%8E%E8%82%A1
#https://www.youtube.com/watch?v=FZ9Kf1xfA40&ab_channel=%E5%BC%82%E7%95%8C%E7%95%AA%E8%8C%84%E8%AF%B4%E7%BE%8E%E8%82%A1
#theory of large numbers maybe each user in discord's call is following a prob distribution
'''
high cred: first to call out stock
		shortest duration
		highest gain
		low number of people call out the same stock
		#returns prob of wining vs prob of losing and the amount to win and lose maxmize profit pass in kelly for each investment interval
		#the formula should mimic the behavior of a sigmoid function where x is the result from the parameters and y is the cred score

		#\frac{6}{\frac{1}{6}+e^{-x}\ }
low:   
'''

#
import logging

class VideoModel(object):
	#uses youtube model with kelly and discord chat faith determination
	def __init__(self):
		pass

	def kelly_formula(self):
		pass


#this user can be removed each user its own table with 
#this allows to see which user have more influence on the stock market price is more accurate


class NLPstock(object):
	def __init__(self, db_name="stocks.db"):
		self.nlp = spacy.load("en_core_web_sm")
		self.db_name = db_name

		self._current_time = datetime.now()
		self._date = self.current_time.date()

	@property
	def current_time(self):
		return self._current_time
		
	@current_time.setter 
	def current_time(self, ct):
		#self.start_hours = ct.replace(hour=9, minute=30, second=0, microsecond=0)
		#self.end_hours = ct.replace(hour=16, minute=00, second=0, microsecond=0)
		
		if (ct.hour >= 5) and (ct.hour < 14):
			self._date = (ct + timedelta(days=-1)).date()
		logging.info(f"setting time.. current hour is {ct.hour}, {self._date} ")

		self._current_time = ct



	def update_stock_table(self, stock_tk, message, c):
		c.execute("SELECT * FROM %s WHERE today = ?" % (stock_tk), (str(self._date),)) 
		rows = c.fetchall()

		logging.info(f"try to fetch for {str(self._date)} stock is {stock_tk} result {rows}")

		if rows:
			c.execute("UPDATE %s SET today_count = today_count + 1 WHERE today = ?" % (stock_tk), (str(self._date),))
			logging.info(f"find existing {str(self._date)} for stock {stock_tk}")
		else:
			#first time of the day
			c.execute('INSERT INTO %s VALUES (?,?,?,?,?)'% (stock_tk), (
				self._date,
				0,
				None,
				message['author']['id'],
				message['timestamp']
			))
			logging.info(f"NO existing {str(self._date)} for stock {stock_tk} creating..... ")



	def insert_stock(self, stock_tk, tk_value, message):
		logging.info(f"inserting stock {stock_tk}.......")
		dbdir = path.join(getcwd(), 'data')
		if not path.exists(dbdir):
			makedirs(dbdir)

		dbfile = path.join(dbdir, self.db_name)
		db = sqlite3.connect(dbfile)
		c = db.cursor()

		c.execute('''CREATE TABLE IF NOT EXISTS stocks (
			ticker TEXT NOT NULL PRIMARY KEY,
			name TEXT,
			count INTEGER,
			call_user TEXT,
			call_price REAL,
			call_time TEXT
		)''')


		c.execute('INSERT INTO %s VALUES (?,?,?,?,?,?)'% ("stocks"), (
			stock_tk,
			tk_value.info['longName'],
			0,
			message['author']['id'],
			tk_value.history('1d')['Close'][0], 
			message['timestamp']
		))

		#when the stock is already made sure to be true
		c.execute('''CREATE TABLE IF NOT EXISTS %s (
			today TEXT NOT NULL PRIMARY KEY,
			today_count INTEGER,
			top_user TEXT,
			first_call TEXT,
			call_time TEXT
		)''' %(stock_tk)) 

		self.update_stock_table(stock_tk, message, c)

		logging.info(f"{stock_tk} Insert Sucess")
		db.commit()
		db.close()


	def stock_in_table(self, stock_tk, message):
		logging.info(f"Finding stock {stock_tk} in tab")

		dbdir = path.join(getcwd(), 'data')
		if not path.exists(dbdir):
			makedirs(dbdir)

		dbfile = path.join(dbdir, self.db_name)
		db = sqlite3.connect(dbfile)
		c = db.cursor()

		c.execute('''CREATE TABLE IF NOT EXISTS stocks (
			ticker TEXT NOT NULL PRIMARY KEY,
			name TEXT,
			count INTEGER,
			call_user TEXT,
			call_price REAL,
			call_time TEXT
		)''')

		
		c.execute("SELECT * FROM stocks WHERE ticker = ?", (stock_tk,)) 
		rows = c.fetchall()
		
		if rows:
			c.execute("UPDATE stocks SET count = count + 1 WHERE ticker = ?", (stock_tk,))
			self.update_stock_table(stock_tk, message, c)
			db.commit()
			db.close()
			return True
		else:
			db.close()
			return False
		

		





	def get_stocks(self, message):
		string_value = message['content']
		self.doc = self.nlp(string_value)
		stock_list = [x.text for x in self.doc.ents if x.label_ == "ORG"]
		stock_list += re.findall("[A-Z]{2,}", string_value)
		stock_list = set(stock_list)


		stock_string = []

		for stock in stock_list:
			processed_stock = self.process_org(stock, message)
			if processed_stock:
				stock_string.append(processed_stock)

		return stock_string





	def process_org(self, stock, message):#for processing the org into a ticker
		stock =stock.strip()
		stock = " ".join(re.findall("[a-zA-Z]+", stock))
		if (len(stock) > 4) or (len(stock) < 2):
			#print(f"Failed: {stock}")
			pass
		else:
			try:
				if self.stock_in_table(stock, message):
					logging.info(f"{stock} already in table")
					return stock

				tk = yf.Ticker(stock)
				#t = threading.Thread()
				self.insert_stock(stock, tk, message)
				return stock

			except KeyError:
				logging.info(f"Yahoo cant find {stock}")
			except Exception as e:
				logging.info(f'Weird stock bugg {stock}')

			#this means either it contains the $ or its not a stock we are looking for




	



if __name__ == "__main__":

	pass
