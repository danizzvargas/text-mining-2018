import numpy as np
import pandas as pd
import time
import pickle
from config import Config

def addLabel(fileInput, dictInput, fileOutput):
	start_time = time.time()

	print("Reading clean file")
	df = pd.read_csv(fileInput,delimiter=',',names=['id', 'article'])

	# Nueva columna que guarda el label de hiperpartidista
	df['hyperpartisan'] = pd.Series(np.zeros(df.shape[0]), index=df.index,dtype = np.int8)

	print("Adding label")

	pickle_in = open(dictInput,"rb")
	idToLabel = pickle.load(pickle_in)
	for index, row in df.iterrows():
		myId = row['id']
		df.set_value(index,'hyperpartisan',idToLabel[myId])

	df.to_csv(fileOutput,header=False,index=False)
	print("Saved")
	print('Total time: %.3f s' % (time.time() - start_time))

def main(args):
  err_msg = 'Unknown function, options: train, validate'
  if len(args) > 1:
    func_name = args[1]
    if func_name == 'train':
      addLabel(Config.FILE_FREQ_ID,"dict.pickle",Config.FILE_TRAIN)
    elif func_name == 'validate':
      addLabel(Config.FILE_FREQ_ID_VAL,"dictVal.pickle",Config.FILE_VAL)
    else:
      print(err_msg)
  else:
    print(err_msg)
  return 0

if __name__ == '__main__':
  import sys
  sys.exit(main(sys.argv))