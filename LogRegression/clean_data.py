"""
Limpieza del corpus. Tiene 2 pasos:

1) 
Obtiene los articulos y su id. 
Limpia los articulos y quita stop words. 
Calcula la frecuencia

Elementos útiles:
    <article>: title, contenido
    <p>      : contenido
    <q>      : contenido
    <a>      : href, contenido

    El resultado sin stop words es guardado en otro archivo temp_articles.csv

2) Carga temp_articles.csv y convierte palabras a orden en frecuencia


"""

import re
import time
import xml.etree.ElementTree as ET
import operator
import pandas as pd
import pickle
from config import Config


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

def main(args):
  err_msg = 'Unknown function, options: train, validate'
  if len(args) > 1:
    func_name = args[1]
    if func_name == 'train':
      clean_process(Config.INPUT_FILE_DATA_TRAIN,Config.FILE_FREQ_ID,Config.MAX_ARTICLES)
    elif func_name == 'validate':
      clean_process(Config.INPUT_FILE_DATA_VAL,Config.FILE_FREQ_ID_VAL,Config.MAX_ARTICLES_VAL)
    else:
      print(err_msg)
  else:
    print(err_msg)
  return 0

def clean_process(input,output,maxA):
    start_time = time.time()

    count = 0

    context = iter(ET.iterparse(input, events=['start','end']))
    _, root = next(context)

    print("Clean and convert")
    with open(output, 'w', encoding='UTF-8') as file:

        article = ''

        countP = 0
        countA = 0
        for event, elem in context:
            # Etiqueta <article>.
            if elem.tag == 'article':

                if event == 'start':                        # Apertura de un artículo.
                    article = elem.attrib['title'] + ' '
                    id = int(elem.attrib['id'])
                    file.write(str(id)+',')
                    countP = 0
                    countA = 0
                else:                                       # Cierre de un artículo.
                    article = clean(article)
                    article = ' '.join([word for word in article.split()])
                    file.write(article+',')
                    file.write(str(countP)+ ',')
                    file.write(str(countA)+ '\n')
                    count+=1
                    if count >= maxA:
                        break

            # Cierre de etiquetas (excepto article).
            elif event == 'end':
                pass

            # Apertura de etiquetas <p>, <q> o <a>.
            elif elem.tag == 'p' or elem.tag == 'q':
                if elem.text != None:
                    article = article + elem.text + ' '
                    countP += 1
            elif elem.tag == 'a':
                if elem.text != None:
                    article = article + elem.text + ' '
                    countA += 1

            root.clear()

if __name__ == '__main__':
  import sys
  sys.exit(main(sys.argv))