import numpy as np
from utils import *
import emoji
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Embedding
from keras.layers import LSTM, Bidirectional
from keras.models import load_model, save_model
from matplotlib import pyplot
from keras.models import Model
dataFrame=pd.read_csv('data/train_emoji.csv')

x=dataFrame.values[:,0]

y=dataFrame.values[:,1]

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

label_encoder,encoded=convert_to_one_hot(y)

x_vectors=[]

for i in x:
	x_vectors.append(sentence_to_avg(i, word_to_vec_map))
	pass

x_vectors=np.array(x_vectors)

x_train,x_val,y_train,y_val=train_test_split(x_vectors,encoded)


print('x train shape')
print(x_train.shape)

print('y train shape')
print(y_train.shape)

#Building model

model = Sequential()

model.add(Dense(y_train.shape[1],input_dim=x_train.shape[1],activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# model.load_weights('models/softmax_weights.h5')

history = model.fit(x_train, y_train, epochs=100,validation_data=(x_val,y_val), batch_size=10, verbose=1)


print('Predicted')

print(label_to_emoji_list(convert_to_labels(model.predict(x_val))))

print('Acutual')

print(label_to_emoji_list(convert_to_labels(y_val)))

model.save('models/softmax.h5')

model.save_weights('models/softmax_weights.h5')

model.summary()

pyplot.plot(history.history['acc'],label='Training Accuracy')

pyplot.plot(history.history['val_acc'],label='Validation Accuracy')

pyplot.legend()

pyplot.show()
