#compute all the count values for each stock to date
#compute credibility score for each user
#https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/

#def cred_compute(stock_name, total_caller, ): #this should behave like x^3 graph, computed daily
import os, sys, glob
import sqlite3
import pandas as pd

def get_pd_table(table_value, con):
	q = "SELECT * FROM %s WHERE stocks != ''" % (table_value)
	#values = (street_name,)
	return pd.read_sql_query(q, con)

dbdir = os.path.join(os.getcwd(), 'Discord Scrapes')
if not os.path.exists(dbdir):
    makedirs(dbdir)

dbfile = os.path.join(dbdir, 'user.db')


con = sqlite3.connect(dbfile)
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables= cursor.fetchall()

merge_count = None

for table in tables:
	df = get_pd_table(table, con)
	count_df = df.groupby(['stocks']).agg(['count'])
	print(count_df)
	break

	#print(df)
	#print(df.describe())
