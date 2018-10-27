"""
Ejemplo tomado de:
	https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/?fbclid=IwAR3EHNzkg_9bIGNw-QErX3cPyLmhSXNOHvE0ErmRnMILNuKMTOKX_VYAsCU


Enfoques:
	Para los datos de entrada:
		Word embeddings para representación de las palabras (se puede evitar capa de embeddings), o
		utilizar frecuencia de palabras para indexar a las mismas (como el ejemplo de IMDB).
		En caso de word embeddings, probar con Word2Vec, FastText y ELMo.
		Se pueden combinar los distintos embeddings.
		¿Usar mayúsculas y minúsculas?
	Para el entrenamiento:
		Utilizar Dropout, tanto en las capas ocultas como en la de entrada.
		Utilizar CNN para extraer las dimensiones más relevantes de los vectores de entrada.
		Cambiar LSTM por GRU.
		¿Utilizar biLM? Ver cómo implementarlo en Keras.

		Dos opciones después del embedding:
			RRN - como en keras_lstm_example.py
			Capa densa - en necesario aplicar Flatten.
				If you wish to connect a Dense layer directly to an Embedding layer, you must first flatten the 2D output matrix to a 1D vector using the Flatten layer.
				https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/


	Comparar tranformación entre embeddings utilizando SQLite y modelo en RAM.
	https://github.com/vzhong/embeddings

	Considerar el espacio en RAM del corpus completo y de los embeddings.
	Se puede utilizar:
		https://github.com/vzhong/embeddings
	para guardar datos en memoria.

	Ejemplos y ligas útiles:
		IMDB:
			https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
		Word2Vec + RNN en Keras:
			https://stackoverflow.com/questions/42064690/using-pre-trained-word2vec-with-lstm-for-word-generation
		
"""

# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000

# Words are indexed by overall frequency in the dataset. V. g. the integer "3" encodes the 3rd most frequent word in the data.
# If the num_words argument was specific, the maximum possible index value is num_words-1.
# v. g. X_train[0] contiene sólo las palabras más frecuentes (de 0 a top_words - 1)
# https://keras.io/datasets/
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
