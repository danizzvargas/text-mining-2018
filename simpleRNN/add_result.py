import numpy as np
import pandas as pd
import time
import pickle
from config import Config

start_time = time.time()

print("Reading clean file")
df = pd.read_csv(Config.FILE_FREQ_ID,delimiter=',',names=['id', 'article'])

# Nueva columna que guarda el label de hiperpartidista
df['hyperpartisan'] = pd.Series(np.zeros(df.shape[0]), index=df.index,dtype = np.int8)

print("Adding label")

pickle_in = open("dict.pickle","rb")
idToLabel = pickle.load(pickle_in)
for index, row in df.iterrows():
	myId = row['id']
	df.set_value(index,'hyperpartisan',idToLabel[myId])

df.to_csv(Config.FILE_TRAIN,header=False,index=False)
print("Saved")
print('Total time: %.3f s' % (time.time() - start_time))