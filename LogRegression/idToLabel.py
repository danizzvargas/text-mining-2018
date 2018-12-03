import xml.etree.ElementTree as ET
import time
import pickle
from config import Config

def id_get(fileInput, fileOutput):
	start_time = time.time()

	idToLabel = {}

	print("Reading Ground Truth")
	context = iter(ET.iterparse(fileInput, events=['start','end']))
	_, root = next(context)

	print("Adding label")
	for event, elem in context:
		# Etiqueta <article>.
		if elem.tag == 'article':
			if event == 'start':
				idTag = int(elem.attrib['id'])
				truth = elem.attrib['hyperpartisan']

				if truth == 'true':
					idToLabel[idTag] = 1
				else:
					idToLabel[idTag] = 0
		elif event == 'end':
			pass

		root.clear()

	pickle_out = open(fileOutput,"wb")
	pickle.dump(idToLabel, pickle_out)
	pickle_out.close()

	print("Saved")
	print('Total time: %.3f s' % (time.time() - start_time))

def main(args):
  err_msg = 'Unknown function, options: train, validate'
  if len(args) > 1:
    func_name = args[1]
    if func_name == 'train':
      id_get(Config.INPUT_FILE_DATA_TRAIN_GROUDTRUTH,"dict.pickle")
    elif func_name == 'validate':
      id_get(Config.INPUT_FILE_DATA_VAL_GROUDTRUTH,"dictVal.pickle")
    elif func_name == 'test':
      id_get(Config.INPUT_FILE_DATA_TEST_GROUDTRUTH,"dictTest.pickle")
    else:
      print(err_msg)
  else:
    print(err_msg)
  return 0

if __name__ == '__main__':
  import sys
  sys.exit(main(sys.argv))