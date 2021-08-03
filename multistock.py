from multiprocessing import Pool, Process, Queue, cpu_count, current_process
from util import *
from findstock import Trades
import logging
from datetime import datetime
import os, sys, glob
import traceback
import random

#https://pymotw.com/2/multiprocessing/communication.html

def create_logger(file=None, logging_level = logging.INFO):
    import multiprocessing, logging
    logger = multiprocessing.get_logger()
    logger.setLevel(logging_level)
    formatter = logging.Formatter(\
        '[%(asctime)s| %(levelname)s| %(processName)s] %(message)s')
    if not file:
    	handler = logging.StreamHandler()
    else:
    	handler = logging.FileHandler(file)
    handler.setFormatter(formatter)

    # this bit will make sure you won't have 
    # duplicated messages in the output
    if not len(logger.handlers): 
        logger.addHandler(handler)
    return logger


def find_stocks(wb):
 	list_gainer=wb.active_gainer_loser(direction='gainer', rank_type='preMarket', count=50)["data"]
 	list_active = wb.active_gainer_loser(direction='active', rank_type='volume', count=50)["data"]
 	list_looser = wb.active_gainer_loser(direction='loser', rank_type='preMarket', count=50)["data"]

 	list_gainer_values = [x["ticker"]["symbol"] for x in list_gainer]
 	list_active_values = [x["ticker"]["symbol"] for x in list_active]
 	list_loser_values = [x["ticker"]["symbol"] for x in list_looser]

 	overlap = list(set(list_gainer_values) & set(list_active_values))

 	while len(overlap) < 10:
 		temp = list_gainer_values.pop()
 		if not temp in overlap:
 			overlap.append(temp)

 	print(overlap)
 	return overlap

class Worker(Process):
	def __init__(self, queue, queue_send, wb, generate_csv=True, logging_file=None):
		super(Worker, self).__init__()
		self.queue = queue
		self.queue_send = queue_send
		self.wb = wb
		self.daemon =True
		self.generate_csv=generate_csv
		self.logging_object = create_logger(logging_file)

	def run(self):
		print('Worker started')
		# do some initialization here
		for data in iter(self.queue.get, None):

			print(data)
			try:
				pdclass = Plotdata(data, self.wb, period="2", timeframe="1", backtest=False, trunc=True, extendedhour=0, gui=False)
				if self.generate_csv:
					self.queue_send.put(pdclass.strat_object.df_result_list)
					#if we dont add stuff in the queue send, then the program will hang...
				else:
					counter = 0
					while True:
						if counter % 1000 ==0:
							print(f"worker for {data} still running...")
						pdclass.strat_object.process_data("1")
						counter +=1

			except Exception as e:
				self.queue_send.put([data, f"{e}", None, None, None, None, None, None, None, None, None, None, None])
				print(f"Process for {data} encountered Error... {e} {traceback.format_exc()}")

			print(f"process for {data} has finished running....")
			

			Trades.starting = 1000
			Trades.instances = []
			Trades.stock_count = 0

if __name__ == "__main__":

	dt = datetime.utcnow().timestamp()
	dir_name = os.path.join('./loggings', str(dt))
	os.mkdir(dir_name)

	print("making dir ", dir_name)
	logging.basicConfig(filename=os.path.join(dir_name, 'output.log'), filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


	#workers_amount = 5

	workers_amount = cpu_count() + 6
	

	df_list = []

	#the_real_source = get_stock_list()[:1000]
	#the_real_source = random.choices(the_real_source, k=10)
	#the_real_source = ["DOGEUSD", "BTCUSD", "ETHUSD", "LTCUSD", "BCHUSD", "XLMUSD", "DASHUSD"]
	#the_real_source = ["DOGEUSD", "BTCUSD", "ETHUSD", "LTCUSD", "BCHUSD", "XLMUSD", "DASHUSD"]

	wb = login()

	the_real_source = find_stocks(wb)

	result_queue = Queue()
	request_queue = Queue()

	workers = []
	print(f"Spawning {workers_amount} of workers...")

	for i in range(workers_amount):
		workers.append(Worker(request_queue, result_queue, wb, generate_csv=True))

	for work in workers:
		work.start()
	for data in the_real_source:
		request_queue.put(data)


	# Sentinel objects to allow clean shutdown: 1 per worker.
	for i in range(workers_amount):
		request_queue.put(None) 

	

	for x in the_real_source:
		df_list.append(result_queue.get())

	for work in workers:
		work.join()

	pd.DataFrame(df_list,columns=["ticker", "start", "final","money","stocks", "precent", "buy count", "sell count", "hold perc", "start time", "start price", "end time", "end price"]).to_csv("result.csv")




