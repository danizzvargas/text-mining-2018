# coding=utf-8
#!/usr/bin/env python

"""Term frequency extractor for the PAN19 hyperpartisan news detection task"""
# Version: 2018-10-09

# Parameters:
# --inputDataset=<directory>
#   Directory that contains the articles XML file with the articles for which a prediction should be made.
# --outputFile=<file>
#   File to which the term frequency vectors will be written. Will be overwritten if it exists.

# Output is one article per line:
# <article id> <token>:<count> <token>:<count> ...


import os
import getopt
import sys
import xml.sax
import lxml.sax
import lxml.etree
import re
import operator
import nltk
from nltk.corpus import stopwords

termfrequencies = {}
stop_words = stopwords.words('english')


########## OPTIONS HANDLING ##########
def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["inputDataset=", "outputFile="]
        opts, _ = getopt.getopt(sys.argv[1:], "d:o:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputDataset = "undefined"
    outputFile = "undefined"

    for opt, arg in opts:
        if opt in ("-d", "--inputDataset"):
            inputDataset = arg
        elif opt in ("-o", "--outputFile"):
            outputFile = arg
        else:
            assert False, "Unknown option."
    if inputDataset == "undefined":
        sys.exit("Input dataset, the directory that contains the articles XML file, is undefined. Use option -d or --inputDataset.")
    elif not os.path.exists(inputDataset):
        sys.exit("The input dataset folder does not exist (%s)." % inputDataset)

    if outputFile == "undefined":
        sys.exit("Output file, the file to which the vectors should be written, is undefined. Use option -o or --outputFile.")

    return (inputDataset, outputFile)

def clean_data(text):
    """ Limpia el texto recibido. """
    text = re.sub(r'\?[a-zA-Z]\b', ' ', text)  # Cambia (?) seguido de una letra por espacio.
    # text = re.sub(r'&amp;', '', text)  # Quita &amp; No es necesario ya que XML lo convierte al símbolo (&).
    text = re.sub(r'#160;', '', text)  # Quita #160;
    text = re.sub(r'[^0-9a-zA-Z\. ]', '', text)  # Quita símbolos especiales. Los puntos pueden utilizarse obtener sentencias.
    text = re.sub(r' {2,}', ' ', text)  # Quita espacios en blanco repetidos.
    return text.strip()

########## ARTICLE HANDLING ##########
def handleArticle(article):

    # get text from article
    text = lxml.etree.tostring(article, method="text")
    textcleaned = re.sub('[^a-z ]', '', text.lower())
    textcleaned = clean_data(textcleaned)

    # counting tokens
    for token in textcleaned.split():
        if token in stop_words:
            continue;
        if token in termfrequencies:
            termfrequencies[token] += 1
        else:
            termfrequencies[token] = 1



########## SAX FOR STREAM PARSING ##########
class HyperpartisanNewsTFExtractor(xml.sax.ContentHandler):
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)
        self.lxmlhandler = "undefined"

    def startElement(self, name, attrs):
        if name != "articles":
            if name == "article":
                self.lxmlhandler = lxml.sax.ElementTreeContentHandler()

            self.lxmlhandler.startElement(name, attrs)

    def characters(self, data):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.characters(data)

    def endElement(self, name):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.endElement(name)
            if name == "article":
                # pass to handleArticle function
                handleArticle(self.lxmlhandler.etree.getroot())
                self.lxmlhandler = "undefined"


########## MAIN ##########
def main(inputDataset, outputFile):
    """Main method of this module."""

    with open(outputFile, 'w') as outFile:
        for file in os.listdir(inputDataset):
            if file.endswith(".xml"):
                with open(inputDataset + "/" + file) as inputRunFile:
                    xml.sax.parse(inputRunFile, HyperpartisanNewsTFExtractor())

        sorted_x = sorted(termfrequencies.items(), key=operator.itemgetter(1))
        
        for token, count in sorted_x:
            outFile.write(" " + str(token) + ":" + str(count))
            outFile.write("\n")

    print("The vectors have been written to the output file.")


if __name__ == '__main__':
    main(*parse_options())

