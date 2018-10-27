"""
Limpieza del corpus.

Elementos útiles:
    <article>: title, contenido
    <p>      : contenido
    <q>      : contenido
    <a>      : href, contenido

Dudas y comentarios:
  * ¿Mezclar todos los contenidos o separarlos en arreglos distintos?
  * Problema. Se reconocen algunos caracteres especiales de HTML como parte del XML (por ejemplo: &amp; ).
        https://en.wikipedia.org/wiki/List_of_XML_and_HTML_character_entity_references#Predefined%5Fentities%5Fin%5FXML
"""

import re
import time
import xml.etree.ElementTree as ET

training_path = '../../data/articles-training-20180831.xml/articles-training-20180831.xml'
training_labels_path = '../../data/ground-truth-training-20180831.xml/ground-truth-training-20180831.xml'

validation_path = '../../data/articles-validation-20180831.xml/articles-validation-20180831.xml'
validation_labels_path = '../../data/ground-truth-validation-20180831.xml/ground-truth-validation-20180831.xml'


####### CONFIGURA DATASET #######

dataset = 'validation'  # 'train' o 'validation'
LIMIT = 100

##################################

if dataset == 'train':
    articles_path  = training_path
    labels_path    = training_labels_path
elif dataset == 'validation':
    articles_path  = validation_path
    labels_path    = validation_labels_path

output_path = 'dataset/{}_{}.txt'.format(dataset, LIMIT)
print('Dataset:', output_path)

##################################

def clean_data(text):
    """ Limpia el texto recibido. """
    text = re.sub(r'\?[a-zA-Z]\b', ' ', text)  # Cambia (?) seguido de una letra por espacio.
    # text = re.sub(r'&amp;', '', text)  # Quita &amp; No es necesario ya que XML lo convierte al símbolo (&).
    text = re.sub(r'#160;', '', text)  # Quita #160;
    text = re.sub(r'[^0-9a-zA-Z\. ]', '', text)  # Quita símbolos especiales. Los puntos pueden utilizarse obtener sentencias.
    text = re.sub(r' {2,}', ' ', text)  # Quita espacios en blanco repetidos.
    return text.strip()


start_time = time.time()


articles_xml = iter(ET.iterparse(articles_path, events=['start','end']))
labels_xml = iter(ET.iterparse(labels_path, events=['start']))

_, root_articles = next(articles_xml)
_, _ = next(labels_xml)  # lee el primer elemento (lo omite).

cnt = 0
with open(output_path, 'w', encoding='UTF-8') as output_file:

    article = ''
    for event, elem in articles_xml:
        # Etiqueta <article>.
        if elem.tag == 'article':

            if event == 'start':                        # Apertura de un artículo.
                cnt = cnt+1
                article = elem.attrib['title'] + ' '
                idx = elem.attrib['id']
            else:                                       # Cierre de un artículo.
                _, label = next(labels_xml)

                if idx != label.attrib['id']:
                    raise ValueError('Los IDs no coinciden: {}, {}'.format(idx, label.attrib['id']))

                val = 1 if 'true' == label.attrib['hyperpartisan'] else 0
                output_file.write('{},{},{}\n'.format(idx, clean_data(article), val))

        # Cierre de etiquetas (excepto article).
        elif event == 'end':
            pass

        # Apertura de etiquetas <p>, <q> o <a>.
        elif elem.tag == 'p' or elem.tag == 'q' or elem.tag == 'a':
            if elem.text != None:
                article = article + elem.text + ' '

        root_articles.clear()

        if cnt > LIMIT:
            break

print('Total time: %.3f s' % (time.time() - start_time))
