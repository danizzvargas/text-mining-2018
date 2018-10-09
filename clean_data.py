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

articles_input = 'SemEval_2019_task_4_Hyperpartisan_News_detection/articles-training-20180831.xml/articles-training-20180831.xml'
articles_output = 'clean_articles.txt'


def clean_data(text):
    """ Limpia el texto recibido. """
    text = re.sub(r'\?[a-zA-Z]\b', ' ', text)  # Cambia (?) seguido de una letra por espacio.
    # text = re.sub(r'&amp;', '', text)  # Quita &amp; No es necesario ya que XML lo convierte al símbolo (&).
    text = re.sub(r'#160;', '', text)  # Quita #160;
    text = re.sub(r'[^0-9a-zA-Z\. ]', '', text)  # Quita símbolos especiales. Los puntos pueden utilizarse obtener sentencias.
    text = re.sub(r' {2,}', ' ', text)  # Quita espacios en blanco repetidos.
    return text.strip()


if __name__ == '__main__':
    start_time = time.time()

    context = iter(ET.iterparse(articles_input, events=['start','end']))
    _, root = next(context)

    cnt = 0  # Contador auxiliar para ver al artículo con ID 0000040.

    with open(articles_output, 'w', encoding='UTF-8') as file:

        article = ''
        for event, elem in context:
            # Etiqueta <article>.
            if elem.tag == 'article':

                if event == 'start':                        # Apertura de un artículo.
                    # cnt = cnt+1
                    article = elem.attrib['title'] + ' '
                else:                                       # Cierre de un artículo.
                    file.write(clean_data(article) + '\n')

            # Cierre de etiquetas (excepto article).
            elif event == 'end':
                pass

            # Apertura de etiquetas <p>, <q> o <a>.
            elif elem.tag == 'p' or elem.tag == 'q' or elem.tag == 'a':
                if elem.text != None:
                    article = article + elem.text + ' '

            root.clear()

            # if cnt == 18:
            #     print('{} {} [{}]'.format(event, elem.tag, elem.text))

    print('Total time: %.3f s' % (time.time() - start_time))
