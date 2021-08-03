import time



from multiprocessing import Pool, Process, Queue, cpu_count, current_process, JoinableQueue, get_logger
from util import *
from findstock import Trades
import logging
from datetime import datetime
import os, sys, glob
import traceback
import random
import threading
from teststrat import *

#https://pymotw.com/2/multiprocessing/communication.html
#https://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python
#https://stackoverflow.com/questions/52094432/best-way-to-wait-for-queue-population-python-multiprocessing

SENTINEL = None
DEFAULT_PRIOR = -1



def get_stream_logger(file=None, level=logging.INFO):
	"""Return logger with configured StreamHandler."""
	stream_logger = logging.getLogger('stream_logger')
	stream_logger.handlers = []
	stream_logger.setLevel(level)
	if not file:
		sh = logging.StreamHandler()
	else:
		sh = logging.FileHandler(file)

	sh.setLevel(level)
	fmt = '[%(asctime)s %(levelname)-8s %(processName)s] --- %(message)s'
	formatter = logging.Formatter(fmt)
	sh.setFormatter(formatter)
	stream_logger.addHandler(sh)

	return stream_logger


'''
def create_logger(file=None, logging_level = logging.DEBUG):
	#import multiprocessing, logging
	logger = get_logger()
	logger.setLevel(logging_level)
	formatter = logging.Formatter(\
		'[%(asctime)s| %(levelname)s| %(processName)s] %(message)s')
	if not file:
		handler = logging.StreamHandler(sys.stdout)
	else:
		handler = logging.FileHandler(file)
	handler.setFormatter(formatter)

	# this bit will make sure you won't have 
	# duplicated messages in the output
	if not len(logger.handlers): 
		logger.addHandler(handler)
	return logger
'''
def find_stocks(wb):
	list_gainer=wb.active_gainer_loser(direction='gainer', rank_type='preMarket', count=50)["data"]
	list_active = wb.active_gainer_loser(direction='active', rank_type='volume', count=50)["data"]
	list_looser = wb.active_gainer_loser(direction='loser', rank_type='preMarket', count=50)["data"]
	list_gainer_values = [x["ticker"]["symbol"] for x in list_gainer]
	list_active_values = [x["ticker"]["symbol"] for x in list_active]
	list_loser_values = [x["ticker"]["symbol"] for x in list_looser]

	overlap = list(set(list_gainer_values) & set(list_active_values))[:10]

	while len(overlap) < 10:
		temp = list_gainer_values.pop()
		if not temp in overlap:
			overlap.append(temp)

	print(overlap)
	return overlap


class Worker(Process):
	def __init__(self, queue, queue_send, wb, generate_csv=True, logging_file=None, prev_hist=False):
		super(Worker, self).__init__()
		self.queue = queue
		self.queue_send = queue_send
		self.wb = wb
		self.daemon =True
		self.generate_csv=generate_csv
		self.logging_file = logging_file
		self.prev_hist = prev_hist #if this is True then, we are using thread in this process to populate the old data
		self.strat = MA2050()

	def thread_function(self, data_value):
		self.logging_object.info(f"getting {data_value} started...")
		#current_price = self.wb.get_bars(stock=data_value, interval='m1', count=1, extendTrading=1)
		current_price = wb.get_bars(stock=data_value, interval='m1', count=100, extendTrading=1)
		self.queue.put((1, data_value, current_price, -1))
		self.logging_object.info(f"getting {data_value} ended...")
		print(f"{data_value}: ${current_price.close}")

	def run(self):
		#the queue passed in needs to be used
		# do some initialization here
		self.logging_object = get_stream_logger(self.logging_file)
		try:
			
			for data in iter(self.queue.get, SENTINEL): #bascially while loop forever until get none
				priority, data_value, current_price, index = data
				
				if priority == DEFAULT_PRIOR:
					#time to get from network start thread	
					threading.Thread(target=self.thread_function, args=(data_value,)).start()
					
				else:
					#after geting from network time to process
					self.logging_object.info(f"Processing {data_value} started...")
					#time.sleep(3)
					self.strat.history_df = current_price
					result = self.strat.buy_sell()
					self.queue_send.put(result.buy)
					self.queue.put((DEFAULT_PRIOR, data_value, None, None))
					self.logging_object.info(f"Processing {data_value} ended...")

		except Exception as e:
			self.logging_object.info(f"process failed {current_process()} {e}")

if __name__ == "__main__":

	dt = datetime.utcnow().timestamp()
	dir_name = os.path.join('./loggings', str(dt))
	os.mkdir(dir_name)

	print("making dir ", dir_name)
	log_file = os.path.join(dir_name, "output.log")


	workers_amount = cpu_count() + 6
	

	df_list = []
	wb = login()

	#the_real_source = get_stock_list()[:1000]
	#the_real_source = random.choices(the_real_source, k=10)
	the_real_source = ["DOGEUSD", "BTCUSD", "ETHUSD", "LTCUSD", "BCHUSD", "XLMUSD", "DASHUSD"]
	x_test = []
	for x in the_real_source:
		x_test.append((DEFAULT_PRIOR, x, None, None))

	the_real_source = x_test
	
	#the_real_source = find_stocks(wb)

	
	request_queue = Queue() #(1, )   (2,)
	result_queue = Queue()
	

	workers = []
	print(f"Spawning {workers_amount} of workers...")

	for i in range(workers_amount):
		workers.append(Worker(request_queue, result_queue, wb, generate_csv=False, logging_file=log_file))

	for work in workers:
		work.start()
	for data in the_real_source:
		request_queue.put(data)


	# Sentinel objects to allow clean shutdown: 1 per worker.
	#for i in range(workers_amount):
	#	request_queue.put(SENTINEL) 

	

	for x in the_real_source:
		df_list.append(result_queue.get())

	for work in workers:
		work.join()



