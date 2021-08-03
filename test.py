from util import *
from finta import TA
import pandas as pd
import numpy as np


wb = login()

count = int((390*int(2))/int(1))

print(count)
df = pd.read_csv("outputtest.csv", index_col="timestamp")

df = wb.get_bars(stock="SENS", interval='m1', count=count, extendTrading=1)
df = df.iloc[-100:, :]


diff = [j-i for i, j in zip(df.index[:-1], df.index[1:])]

print(df)
print(TA.RSI(df))

print(diff)



#df.to_csv("outputtest.csv")