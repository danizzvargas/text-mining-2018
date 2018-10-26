import xml.etree.ElementTree as ET
import time
import pickle
from config import Config

start_time = time.time()

idToLabel = {}

print("Reading Ground Truth")
context = iter(ET.iterparse(Config.INPUT_FILE_DATA_TRAIN_GROUDTRUTH, events=['start','end']))
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

pickle_out = open("dict.pickle","wb")
pickle.dump(idToLabel, pickle_out)
pickle_out.close()

print("Saved")
print('Total time: %.3f s' % (time.time() - start_time))