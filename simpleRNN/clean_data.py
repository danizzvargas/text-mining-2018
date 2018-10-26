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
import nltk
from nltk.corpus import stopwords
import operator
import pandas as pd
from config import Config

articles_temp = 'temp_articles.csv'

count = 0
stop_words = stopwords.words('english')
termfrequencies = {}
termToId = {}

def clean(text):
    """ Limpia el texto recibido. """
    text = re.sub(r'\?[a-zA-Z]\b', ' ', text)  # Cambia (?) seguido de una letra por espacio.
    # text = re.sub(r'&amp;', '', text)  # Quita &amp; No es necesario ya que XML lo convierte al símbolo (&).
    text = re.sub(r'#160;', '', text)  # Quita #160;
    text = re.sub(r'[^0-9a-zA-Z\. ]', '', text)  # Quita símbolos especiales. Los puntos pueden utilizarse obtener sentencias.
    text = re.sub(r' {2,}', ' ', text)  # Quita espacios en blanco repetidos.
    text = text.strip()
    return text

def change_term_id(text):
    article = [str(termToId[word]) for word in text.split() if termToId[word]<Config.TOP_WORDS]
    return article

def getTermToId():
    sorted_x = sorted(termfrequencies.items(), key=operator.itemgetter(1),reverse=True)
    count = 0
    for (term,_) in sorted_x:
        termToId[term]=count
        count+=1

def cleanAndCheckFrequency(text):
    text = clean(text)
    for token in text.split():
        if token not in stop_words:
            if token in termfrequencies:
                termfrequencies[token] += 1
            else:
                termfrequencies[token] = 1
    
    article = ' '.join([word for word in text.split() if word not in stop_words])
    return article


if __name__ == '__main__':
    start_time = time.time()

    context = iter(ET.iterparse(Config.INPUT_FILE_DATA_TRAIN, events=['start','end']))
    _, root = next(context)

    print("Clean and get frequency")
    with open(articles_temp, 'w', encoding='UTF-8') as file:

        article = ''
        for event, elem in context:
            # Etiqueta <article>.
            if elem.tag == 'article':

                if event == 'start':                        # Apertura de un artículo.
                    article = elem.attrib['title'] + ' '
                    id = int(elem.attrib['id'])
                    file.write(str(id)+',')
                else:                                       # Cierre de un artículo.
                    file.write(cleanAndCheckFrequency(article) + '\n')
                    count+=1
                    if count >= Config.MAX_ARTICLES:
                        break

            # Cierre de etiquetas (excepto article).
            elif event == 'end':
                pass

            # Apertura de etiquetas <p>, <q> o <a>.
            elif elem.tag == 'p' or elem.tag == 'q' or elem.tag == 'a':
                if elem.text != None:
                    article = article + elem.text + ' '

            root.clear()

    print("Convert to term to id")
    getTermToId()

    df = pd.read_csv(articles_temp,delimiter=',',names=['id', 'article'])

    for index, row in df.iterrows():
        text=change_term_id(row['article'])
        df.set_value(index,'article',text)

    df.to_csv(Config.FILE_FREQ_ID,header=False,index=False)
    print('Total time: %.3f s' % (time.time() - start_time))
