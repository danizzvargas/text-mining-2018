import re
import time
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

from config import Config

stop_words = stopwords.words('english')

def clean(text):
    """ Limpia el texto recibido. """
    text = re.sub(r'\?[a-zA-Z]\b', ' ', text)  # Cambia (?) seguido de una letra por espacio.
    # text = re.sub(r'&amp;', '', text)  # Quita &amp; No es necesario ya que XML lo convierte al símbolo (&).
    text = re.sub(r'#160;', '', text)  # Quita #160;
    text = re.sub(r'[^0-9a-zA-Z\. ]', '', text)  # Quita símbolos especiales. Los puntos pueden utilizarse obtener sentencias.
    text = re.sub(r' {2,}', ' ', text)  # Quita espacios en blanco repetidos.
    text = text.strip()
    text = text.lower()
    return text

start_time = time.time()

context = iter(ET.iterparse(Config.INPUT_FILE_DATA_VAL, events=['start','end']))
_, root = next(context)

dictPal={}
for event, elem in context:
	# Etiqueta <article>.
	if elem.tag == 'article':
		if event == 'start':                        # Apertura de un artículo.
			article = elem.attrib['title'] + ' '
		else:                                       # Cierre de un artículo.
			article = clean(article)
			for word in article.split():
				if word not in dictPal and word not in stop_words:
					dictPal[word]=1
			

	# Cierre de etiquetas (excepto article).
	elif event == 'end':
		pass

	# Apertura de etiquetas <p>, <q> o <a>.
	elif elem.tag == 'p' or elem.tag == 'q' or elem.tag == 'a':
		if elem.text != None:
			article = article + elem.text + ' '

	root.clear()
print(len(dictPal))