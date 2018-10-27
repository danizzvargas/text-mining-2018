"""
Basado en:
    https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
"""

import pandas
import gensim
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.preprocessing import sequence

np.random.seed(7)


###### WORD EMBEDDINGS ######

# Carga embeddings.
print('cargando word embeddings...')
model_path = '../../../embeddings/Word2Vec.bin'
embedding_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

# Configura carga de dataset.
training_size   = 10000
validation_size = 1000
training_path   = 'dataset/train_{}.txt'.format(training_size)
validation_path = 'dataset/validation_{}.txt'.format(validation_size)

# Carga dataset.
print('cargando dataset y configurando embeddings...')
training = pandas.read_csv(training_path, header=None, usecols=[1,2])
docs = np.array(training[1])
y_train = np.array(training[2])

# Configura tokenizer.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)

# Recupera tamaño de vocabulario y el de los embeddings.
vocab_size = len(tokenizer.word_index) + 1
embedding_size = embedding_model.vector_size

# Codifica los documentos como arreglos de enteros.
encoded_docs = tokenizer.texts_to_sequences(docs)

# Aplica padding (para entrenamiento).
max_length = 500
x_train = sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# Construye una matriz de pesos de tamaño [ vocab_size x embedding_size ].
# Asume que el tamaño máximo de un vector es igual al tamaño del vocabulario.
embedding_matrix = np.zeros((vocab_size, embedding_size))
print('Forma de embedding_matrix:', embedding_matrix.shape)

# Construye matriz de embeddings.
print('Construyendo matriz de pesos (embedding matrix)...')
for word, i in tokenizer.word_index.items():
    if word in embedding_model:
        embedding_matrix[i] = embedding_model[word]


###### RED NEURONAL ######

model = Sequential()
embedding_layer = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], input_length=max_length, trainable=False)
model.add(embedding_layer)
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
# scores = model.evaluate(x_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
