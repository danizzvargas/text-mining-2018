# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Dropout,GRU, BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from ast import literal_eval
from config import Config
from matplotlib import pyplot
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras import regularizers

# fix random seed for reproducibility
np.random.seed(7)

df = pd.read_csv('train.csv',delimiter=',',names=['id', 'article','hyperpartisan'])
df['article'] = df['article'].apply(literal_eval)

df = df.values
X_train = df[:,1]
X_train = [map(int, article) for article in X_train]
y_train = df[:,2].astype('int')
print(len(X_train))
print(y_train.shape)

df = pd.read_csv('validate.csv',delimiter=',',names=['id', 'article','hyperpartisan'])
df['article'] = df['article'].apply(literal_eval)

df = df.values
X_test = df[:,1]
X_test = [map(int, article) for article in X_test]
y_test= df[:,2].astype('int')

pickle_in = open("termToId.pickle","rb")
termToId = pickle.load(pickle_in)

embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	if word in termToId:
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[termToId[word]] = coefs

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((Config.TOP_WORDS, Config.EMBEDDING_VECTOR_LEN))
for term,idTerm in termToId.iteritems():
	if idTerm in embeddings_index:
		embedding_matrix[idTerm] = embeddings_index[idTerm]

X_train = sequence.pad_sequences(X_train, maxlen=Config.MAX_REVIEW_LEN)
X_test = sequence.pad_sequences(X_test, maxlen=Config.MAX_REVIEW_LEN)
# create the model
model = Sequential()
model.add(Embedding(Config.TOP_WORDS, Config.EMBEDDING_VECTOR_LEN, weights=[embedding_matrix], trainable=False))
model.add(GRU(20,dropout=0.2, recurrent_dropout=0.2, activation=None, kernel_regularizer=regularizers.l2(0.05),
	recurrent_regularizer=regularizers.l2(0.05)))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, epochs=10,shuffle=True, batch_size=64,validation_data=(X_test, y_test))

predictions = model.predict(X_test).round().astype('int')
print('Accuracy {}'.format(accuracy_score(y_test,predictions.round())))
print('Precision {}'.format(precision_score(y_test,predictions.round())))
print('Recall {}'.format(recall_score(y_test,predictions.round())))
print('F1 {}'.format(f1_score(y_test,predictions.round())))
# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('model train vs validation accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()