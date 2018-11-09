# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from ast import literal_eval
from config import Config
from matplotlib import pyplot

# fix random seed for reproducibility
np.random.seed(7)

df = pd.read_csv('train.csv',delimiter=',',names=['id', 'article','hyperpartisan'])
df['article'] = df['article'].apply(literal_eval)

df = df.values
X_train = df[:,1]
X_train = [map(int, article) for article in X_train]
y_train = df[:,2]
print(len(X_train))
print(y_train.shape)

df = pd.read_csv('validate.csv',delimiter=',',names=['id', 'article','hyperpartisan'])
df['article'] = df['article'].apply(literal_eval)

df = df.values
X_test = df[:,1]
X_test = [map(int, article) for article in X_test]
y_test= df[:,2]
print(len(X_test))
print(y_test.shape)


X_train = sequence.pad_sequences(X_train, maxlen=Config.MAX_REVIEW_LEN)
X_test = sequence.pad_sequences(X_test, maxlen=Config.MAX_REVIEW_LEN)
# create the model
model = Sequential()
model.add(Embedding(Config.TOP_WORDS, Config.EMBEDDING_VECTOR_LEN, input_length=Config.MAX_REVIEW_LEN))
model.add(LSTM(50,return_sequences=True,dropout=0.5, recurrent_dropout=0.5))
model.add(LSTM(50,dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, epochs=5, batch_size=64,validation_data=(X_test, y_test))
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